# Part of the code adopted from DualSDF repository.
import torch
import numpy as np
import torch.nn as nn
import os
import trimesh
from dgcnn import DGCNNFeat


def bsmin(a, dim, k=22.0, keepdim=False):
    dmix = -torch.logsumexp(-k * a, dim=dim, keepdim=keepdim) / k
    return dmix


def determine_superquadric_sdf(query_points, superquadric_params):
    """Query superquadric sdf for a set of points.

    Args:
        query_points (torch.tensor): Nx3 tensor of query points.
        superquadric_params (torch.tensor): Kx8 tensor of superquadric parameters (center, scale, exponents).

    Returns:
        torch.tensor: Signed distance field of each superquadric primitive with respect to each query point. NxK tensor.
    """
    # Extract parameters
    centers = superquadric_params[:, :3]      # K x 3
    scales = superquadric_params[:, 3:6]      # K x 3
    exponents = superquadric_params[:, 6:8]   # K x 2

    # Expand dimensions for pairwise computation
    query_points_expanded = query_points.unsqueeze(1)      # N x 1 x 3
    centers_expanded = centers.unsqueeze(0)                # 1 x K x 3
    scales_expanded = scales.unsqueeze(0)                  # 1 x K x 3
    exponents_expanded = exponents.unsqueeze(0)            # 1 x K x 2

    # Compute normalized distances
    diff = (query_points_expanded - centers_expanded) / scales_expanded  # N x K x 3
    abs_diff = torch.abs(diff) + 1e-6  # Prevent division by zero

    # Compute superquadric distance
    term1 = (abs_diff[..., 0] ** (2 / exponents_expanded[:, :, 0])) + \
            (abs_diff[..., 1] ** (2 / exponents_expanded[:, :, 0])) + \
            (abs_diff[..., 2] ** (2 / exponents_expanded[:, :, 0]))
    sdf = (term1 ** (exponents_expanded[:, :, 1] / 2) - 1) * torch.min(scales_expanded, dim=-1)[0]

    return sdf  # N x K


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        in_ch = 256
        out_ch = 256 * 8  # 8 parameters per superquadric
        feat_ch = 512

        self.net1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch - in_ch)),
            nn.ReLU(inplace=True),
        )

        self.net2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
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


class SQNet(nn.Module):
    def __init__(self, num_superquadrics=256):
        super(SQNet, self).__init__()
        self.num_superquadrics = num_superquadrics
        self.encoder = DGCNNFeat(global_feat=True)
        self.decoder = Decoder()

    def forward(self, surface_points, query_points):
        features = self.encoder(surface_points)
        superquadric_params = self.decoder(features)
        
        superquadric_params = torch.sigmoid(superquadric_params.view(-1, 8))  # Changed to 8
        superquadric_adder = torch.tensor([-0.5, -0.5, -0.5, 0.1, 0.1, 0.1, 0.2, 0.2]).to(superquadric_params.device)
        superquadric_multiplier = torch.tensor([1.0, 1.0, 1.0, 0.4, 0.4, 0.4, 0.8, 0.8]).to(superquadric_params.device)
        superquadric_params = superquadric_params * superquadric_multiplier + superquadric_adder
        ### End of changes ###
        
        superquadric_sdf = determine_superquadric_sdf(query_points, superquadric_params)
        return superquadric_sdf, superquadric_params


def visualise_superquadrics(superquadric_params, reference_model, save_path=None):
    superquadric_params = superquadric_params.cpu().detach().numpy()
    centers = superquadric_params[..., :3]
    scales = superquadric_params[..., 3:6]
    exponents = superquadric_params[..., 6:8]
    scene = trimesh.Scene()
    if reference_model is not None:
        scene.add_geometry(reference_model)
    for center, scale, exponent in zip(centers, scales, exponents):
        # Create a superquadric mesh
        # Note: Trimesh does not have a built-in superquadric creator,
        # so we'll approximate it using scaled icospheres for simplicity
        superquadric = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        superquadric.apply_scale(scale)
        superquadric.apply_translation(center)
        scene.add_geometry(superquadric)
    if save_path is not None:
        scene.export(save_path)
    scene.show()


def visualise_sdf(points, values):
    """Visualise the SDF values as a point cloud."""
    # Use trimesh to create a point cloud from the SDF values
    inside_points = points[values < 0]
    outside_points = points[values > 0]
    inside_points = trimesh.points.PointCloud(inside_points, colors=[[0, 0, 255, 255]])  
    outside_points = trimesh.points.PointCloud(outside_points, colors=[[255, 0, 0, 255]])  

    scene = trimesh.Scene()
    scene.add_geometry([inside_points, outside_points])
    scene.show()


def main():
    dataset_root_path = "./reference_models_processed"  # New root directory for datasets
    object_names = ["dog", "hand", "pot", "rod", "sofa"]  # Names of the objects
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 1000

    for obj_name in object_names:
        print(f"Processing object: {obj_name}")
        
        object_path = os.path.join(dataset_root_path, obj_name)
        try:
            # Load mesh model and point cloud
            mesh_model = trimesh.load(os.path.join(object_path, f"{obj_name}.obj"))
            pcd_model = trimesh.load(os.path.join(object_path, "surface_points.ply"))
            
            # Load point and SDF values
            data_npz = np.load(os.path.join(object_path, "voxel_and_sdf.npz"))
            points, values = data_npz["sdf_points"], data_npz["sdf_values"]
            
            # Convert to torch tensors
            points = torch.from_numpy(points).float().to(device)
            values = torch.from_numpy(values).float().to(device)
            surface_pointcloud = torch.from_numpy(pcd_model.vertices).float().to(device)
            
            # Initialize model and optimizer
            model = SQNet(num_superquadrics=256).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
            
            # Training loop
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                superquadric_sdf, superquadric_params = model(
                    surface_pointcloud.unsqueeze(0).transpose(2, 1), points
                )
                # Smooth minimum to combine SDFs
                superquadric_sdf = bsmin(superquadric_sdf, dim=-1).to(device)
                
                # Loss function: Mean squared error
                mseloss = torch.mean((superquadric_sdf - values) ** 2)
                loss = mseloss
                
                loss.backward()
                optimizer.step()
                
                if epoch % 100 == 0:  # Print every 50 epochs
                    print(f"[{obj_name}] Epoch {epoch}, Loss: {loss.item()}")

            # Save the results
            output_dir = os.path.join("./output", obj_name)
            os.makedirs(output_dir, exist_ok=True)
            superquadric_params_np = superquadric_params.cpu().detach().numpy()
            np.save(os.path.join(output_dir, f"{obj_name}_superquadric_params.npy"), superquadric_params_np)
            
            # Visualize superquadrics
            visualise_superquadrics(
                superquadric_params, reference_model=pcd_model, 
                save_path=os.path.join(output_dir, f"{obj_name}_superquadrics.obj")
            )
            print(f"Finished processing {obj_name}. Results saved in {output_dir}.")
        
        except Exception as e:
            print(f"Error processing {obj_name}: {e}")

if __name__ == "__main__":
    main()
