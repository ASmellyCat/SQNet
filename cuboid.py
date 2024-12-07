import torch
import numpy as np
import torch.nn as nn
import os
import trimesh
from dgcnn import DGCNNFeat
from eval import calculate_metrics

# ===========================
# Modifiable Hyperparameters
# ===========================
num_epochs = 1000
num_cuboids = 4
learning_rate = 0.0005
bsmin_k = 22
coverage_weight = 0
rotation_weight = 0.1
num_surface_points = 1000  # Number of points to sample per cuboid surface
dataset_root_path = "./reference_models_processed"  # Root directory for the dataset
# object_names = ["dog", "hand", "pot", "rod", "sofa"]  # List of object names to process
object_names = [ "sofa"]
output_dir = "./output"
# ===========================

def bsmin(a, dim, k=bsmin_k, keepdim=False): 
    """
    Smooth minimum function for better blending between cuboids.
    Lower k value means smoother transitions between cuboids.
    """
    dmix = -torch.logsumexp(-k * a, dim=dim, keepdim=keepdim) / k
    return dmix

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion to a rotation matrix.
    Args:
        q: tensor of shape (..., 4), where the last dimension represents (w, x, y, z)
    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    # Normalize the quaternion
    q = q / q.norm(p=2, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    B = q.shape[:-1]

    # Compute rotation matrix elements
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z

    xy = x * y
    xz = x * z
    yz = y * z

    rot = torch.stack([
        ww + xx - yy - zz, 2 * (xy - wz),     2 * (xz + wy),
        2 * (xy + wz),     ww - xx + yy - zz, 2 * (yz - wx),
        2 * (xz - wy),     2 * (yz + wx),     ww - xx - yy + zz
    ], dim=-1).reshape(*B, 3, 3)

    return rot

def determine_cuboid_sdf(query_points, cuboid_params):
    """Compute the SDF between query points and cuboids.

    Args:
        query_points (torch.tensor): Nx3 tensor of query points.
        cuboid_params (torch.tensor): Kx10 tensor of cuboid parameters (center, quaternion, dimensions).

    Returns:
        torch.tensor: Signed distance field of each cuboid primitive with respect to each query point. NxK tensor.
    """
    # Extract cuboid parameters
    cuboid_centers = cuboid_params[:, :3]       # K x 3
    cuboid_quaternions = cuboid_params[:, 3:7]  # K x 4
    cuboid_dimensions = cuboid_params[:, 7:]    # K x 3 (dx, dy, dz)

    # Normalize quaternions
    cuboid_quaternions = cuboid_quaternions / torch.norm(cuboid_quaternions, dim=1, keepdim=True)

    # Compute rotation matrices
    rotation_matrices = quaternion_to_rotation_matrix(cuboid_quaternions)  # K x 3 x 3
    rotation_matrices_inv = rotation_matrices.transpose(1, 2)  # Inverse rotation

    # Expand dimensions for broadcasting
    query_points_expanded = query_points.unsqueeze(1)  # N x 1 x 3
    cuboid_centers_expanded = cuboid_centers.unsqueeze(0)  # 1 x K x 3

    # Translate points to cuboid's local frame
    local_points = query_points_expanded - cuboid_centers_expanded  # N x K x 3

    # Rotate points to align with cuboid's axes
    local_points = torch.einsum('nki,kij->nkj', local_points, rotation_matrices_inv)

    # Compute SDF for axis-aligned box centered at origin
    half_dims = cuboid_dimensions.unsqueeze(0) / 2  # 1 x K x 3
    q = torch.abs(local_points) - half_dims  # N x K x 3

    outside_distance = torch.norm(torch.clamp(q, min=0.0), dim=2)
    inside_distance = torch.clamp(torch.max(q, dim=2)[0], max=0.0)
    sdf = outside_distance + inside_distance  # N x K

    return sdf

class Decoder(nn.Module):
    def __init__(self, num_cuboids=num_cuboids):
        super(Decoder, self).__init__()
        self.num_cuboids = num_cuboids
        in_ch = 256
        feat_ch = 512
        out_ch = num_cuboids * 10  # 10 parameters per cuboid

        self.net1 = nn.Sequential(
            nn.utils.parametrizations.weight_norm(nn.Linear(in_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.parametrizations.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.parametrizations.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.parametrizations.weight_norm(nn.Linear(feat_ch, feat_ch - in_ch)),
            nn.ReLU(inplace=True),
        )

        self.net2 = nn.Sequential(
            nn.utils.parametrizations.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.parametrizations.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.parametrizations.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.parametrizations.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.Linear(feat_ch, out_ch),
        )
        num_params = sum(p.numel() for p in self.parameters())
        print("[num parameters: {}]".format(num_params))

    def forward(self, z):
        in1 = z
        out1 = self.net1(in1)
        in2 = torch.cat([out1, in1], dim=-1)
        out2 = self.net2(in2)
        return out2

class CuboidNet(nn.Module):
    def __init__(self, num_cuboids=num_cuboids):
        super(CuboidNet, self).__init__()
        self.num_cuboids = num_cuboids
        self.encoder = DGCNNFeat(global_feat=True)
        self.decoder = Decoder(num_cuboids=num_cuboids)

    def forward(self, surface_points, query_points):
        features = self.encoder(surface_points)
        cuboid_params = self.decoder(features)

        # Reshape and process cuboid parameters
        cuboid_params = cuboid_params.view(-1, 10)  # num_cuboids x 10

        # Process positions
        cuboid_centers = torch.sigmoid(cuboid_params[:, :3])
        center_adder = torch.tensor([-1.0, -1.0, -1.0]).to(cuboid_centers.device)
        center_multiplier = torch.tensor([2.0, 2.0, 2.0]).to(cuboid_centers.device)
        cuboid_centers = cuboid_centers * center_multiplier + center_adder

        # Process quaternions
        cuboid_quaternions = cuboid_params[:, 3:7]
        cuboid_quaternions = cuboid_quaternions / torch.norm(cuboid_quaternions, dim=1, keepdim=True)

        # Process dimensions
        cuboid_dimensions = torch.sigmoid(cuboid_params[:, 7:])
        dimension_adder = torch.tensor([0.1, 0.1, 0.1]).to(cuboid_dimensions.device)
        dimension_multiplier = torch.tensor([2.0, 2.0, 2.0]).to(cuboid_dimensions.device)
        cuboid_dimensions = cuboid_dimensions * dimension_multiplier + dimension_adder

        # Combine processed parameters
        cuboid_params = torch.cat([cuboid_centers, cuboid_quaternions, cuboid_dimensions], dim=1)

        # Compute SDF between query points and cuboids
        cuboid_sdf = determine_cuboid_sdf(query_points, cuboid_params)
        return cuboid_sdf, cuboid_params
    
def compute_coverage_loss(cuboid_params, surface_points):
    """
    Compute coverage loss following the paper's formulation:
    L1({(zm, qm, tm)}, O) = Ep~S(O)||C(p; ∪Pm)||²

    Args:
        cuboid_params: tensor of cuboid parameters (centers, quaternions, dimensions)
        surface_points: points sampled from the object surface (representing S(O))
    """
    # Compute SDF for surface points
    sdf = determine_cuboid_sdf(surface_points, cuboid_params)
    min_distances = bsmin(sdf, dim=-1)
    
    # Only consider points outside the predicted shape
    min_distances_clamped = torch.clamp(min_distances, min=0.0)
    
    # Compute coverage loss without weighting
    coverage_loss = torch.mean(min_distances_clamped ** 2)
    
    return coverage_loss



 
def visualise_cuboids(cuboid_params, reference_model, save_path=None):
    """
    Visualize the cuboids with distinct colors and optionally save the visualization.

    Args:
        cuboid_params (torch.tensor): Kx10 tensor of cuboid parameters.
        reference_model (trimesh.Trimesh): Reference mesh model to display alongside cuboids.
        save_path (str, optional): Path to save the visualized cuboids.
    """
    cuboid_params = cuboid_params.cpu().detach().numpy()
    cuboid_centers = cuboid_params[..., :3]
    cuboid_quaternions = cuboid_params[..., 3:7]
    cuboid_dimensions = cuboid_params[..., 7:]

    # Define a list of distinct colors (RGBA)
    colors = [
        [179, 0, 30, 255],    # Red
        [179, 255, 25, 255],    # Green
        [255, 224, 58, 255],  # Yellow
        [128, 170, 255, 255],  # Cyan
        [255, 102, 255, 255],  # Magenta
        [255, 119, 51, 255],  # Orange
        [196, 77, 255, 255],  # Purple
        [179, 242, 255, 255],  # Teal
        [128, 128, 0, 255],  # Olive
    ]

    scene = trimesh.Scene()
    if reference_model is not None:
        if isinstance(reference_model, trimesh.points.PointCloud):
            # Assign a blue color to the reference point cloud
            reference_model.colors = [[219, 204, 188, 255]] * len(reference_model.vertices)
        scene.add_geometry(reference_model)

    for i, (center, quaternion, dimensions) in enumerate(zip(cuboid_centers, cuboid_quaternions, cuboid_dimensions)):
        box = trimesh.creation.box(extents=dimensions)
        transform = trimesh.transformations.quaternion_matrix(quaternion)
        transform[:3, 3] = center
        box.apply_transform(transform)

        # Assign color from the list, cycling if necessary
        color = colors[i % len(colors)]
        box.visual.face_colors = color

        scene.add_geometry(box)

    if save_path is not None:
        scene.export(save_path)
    scene.show()

def main():
    is_training = False
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
    os.makedirs(output_dir, exist_ok=True)
    print(device)

    for obj_name in object_names:
        print(f"Processing object: {obj_name}")
        
        object_path = os.path.join(dataset_root_path, obj_name)
        mesh_model = trimesh.load(os.path.join(object_path, f"{obj_name}.obj"))
        if isinstance(mesh_model, list):
            print(f"{obj_name}.obj contains multiple meshes. Selecting the first one.")
            mesh_model = mesh_model[0]  # Select the first mesh
        
        pcd_model = trimesh.load(os.path.join(object_path, "surface_points.ply"))
        if isinstance(pcd_model, list):
            print(f"surface_points.ply for {obj_name} contains multiple point clouds. Selecting the first one.")
            pcd_model = pcd_model[0]  # Select the first point cloud
        data_npz = np.load(os.path.join(object_path, "voxel_and_sdf.npz"))
        points, values = data_npz["sdf_points"], data_npz["sdf_values"]
        points = torch.from_numpy(points).float().to(device)
        values = torch.from_numpy(values).float().to(device)
        surface_pointcloud = torch.from_numpy(pcd_model.vertices).float().to(device)

        if is_training: # train the model
            model = CuboidNet(num_cuboids).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                cuboid_sdf, cuboid_params = model(
                    surface_pointcloud.unsqueeze(0).transpose(2, 1), points
                )

                # Main SDF loss
                cuboid_sdf = bsmin(cuboid_sdf, dim=-1, k=bsmin_k).to(device)  
                mse_loss = torch.mean((cuboid_sdf - values) ** 2)

                # Coverage loss
                coverage_loss = compute_coverage_loss(cuboid_params, surface_pointcloud)




                # Quaternion regularization
                quaternions = cuboid_params[:, 3:7]
                rotation_loss = torch.mean(torch.abs(quaternions[:, 1:]))

                # Combined loss with weights
                loss = mse_loss + \
                    rotation_weight * rotation_loss + \
                    coverage_weight * coverage_loss 


                loss.backward()
                optimizer.step()

                if (epoch + 1) % 50 == 0 or epoch == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}, "
                        f"Total Loss: {loss.item():.6f}, "
                        f"MSE Loss: {mse_loss.item():.6f}, "
                        f"Coverage Loss: {coverage_loss.item():.6f}, "
                        f"Rotation Loss: {rotation_loss.item():.6f}")
            # Save the cuboid parameters
            np.save(os.path.join(output_dir, f"{obj_name}_cuboid_params.npy"), cuboid_params.cpu().detach().numpy())

            print("Final Cuboid Parameters:")
            print(cuboid_params)
        else: # load the model
            cuboid_params = torch.from_numpy(np.load(os.path.join(output_dir,obj_name, f"{obj_name}_cuboid_params.npy"))).to(device)

        calculate_metrics(surface_pointcloud.cpu().numpy(), cuboid_params, num_surface_points)
        # Visualize the cuboids with distinct colors
        visualise_cuboids(cuboid_params, reference_model=pcd_model, save_path=os.path.join(output_dir, f"{obj_name}_cuboids.obj"))

if __name__ == "__main__":
    main()
