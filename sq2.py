import torch
import numpy as np
import torch.nn as nn
import os
import traceback
import trimesh
from dgcnn import DGCNNFeat
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F



def bsmin(a, dim, k=22.0, keepdim=False):
    """
    Compute the smooth minimum across a specified dimension using the smooth minimum (bsmin) function.

    Args:
        a (torch.Tensor): Input tensor.
        dim (int): Dimension along which to compute the smooth minimum.
        k (float, optional): Smoothness parameter. Defaults to 22.0.
        keepdim (bool, optional): Whether to retain the dimension. Defaults to False.

    Returns:
        torch.Tensor: Tensor after applying the smooth minimum.
    """
    dmix = -torch.logsumexp(-k * a, dim=dim, keepdim=keepdim) / k
    return dmix


def determine_superquadric_sdf(query_points, superquadric_params):
    """
    Compute the Signed Distance Function (SDF) for a set of query points with respect to multiple superquadrics.

    Args:
        query_points (torch.Tensor): Tensor of shape (N, 3) representing N query points in 3D space.
        superquadric_params (torch.Tensor): Tensor of shape (1, K, 11) representing the parameters
                                           of K superquadrics for each sample in the batch.
                                           Parameters per superquadric:
                                           - 3 for translation (tx, ty, tz)
                                           - 3 for rotation (Euler angles: rx, ry, rz)
                                           - 3 for size (alpha1, alpha2, alpha3)
                                           - 2 for shape (epsilon1, epsilon2)

    Returns:
        torch.Tensor: Tensor of shape (N, K) containing the SDF values of each query point with respect to each superquadric.
    """
    # Extract parameters and remove batch dimension
    translations = superquadric_params[:, :, :3].squeeze(0)      # Shape: (K, 3)
    rotations = superquadric_params[:, :, 3:6].squeeze(0)        # Shape: (K, 3)
    sizes = superquadric_params[:, :, 6:9].squeeze(0)            # Shape: (K, 3)
    exponents = superquadric_params[:, :, 9:11].squeeze(0)       # Shape: (K, 2)

    # Number of query points and superquadrics
    N = query_points.shape[0]  
    K = translations.shape[0]  

    # Expand query points to [N, K, 3]
    query_points_expanded = query_points.unsqueeze(1).repeat(1, K, 1)  # Shape: (N, K, 3)

    # Expand superquadric parameters to match query points
    translations_expanded = translations.unsqueeze(0).repeat(N, 1, 1)    # Shape: (N, K, 3)
    sizes_expanded = sizes.unsqueeze(0).repeat(N, 1, 1)                  # Shape: (N, K, 3)
    exponents_expanded = exponents.unsqueeze(0).repeat(N, 1, 1)          # Shape: (N, K, 2)
    rotations_expanded = rotations.unsqueeze(0).repeat(N, 1, 1)          # Shape: (N, K, 3)

    # Compute difference vectors
    diff = query_points_expanded - translations_expanded  # Shape: (N, K, 3)

    # Generate rotation matrices from Euler angles
    angles = rotations_expanded  # Shape: (N, K, 3)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # Create rotation matrices around X-axis
    R_x = torch.stack([
        torch.ones_like(cos[..., 0]),
        torch.zeros_like(cos[..., 0]),
        torch.zeros_like(cos[..., 0]),
        torch.zeros_like(cos[..., 0]),
        cos[..., 0], -sin[..., 0],
        torch.zeros_like(cos[..., 0]),
        sin[..., 0], cos[..., 0]
    ], dim=-1).reshape(-1, 3, 3)  # Shape: (N * K, 3, 3)

    # Create rotation matrices around Y-axis
    R_y = torch.stack([
        cos[..., 1], torch.zeros_like(cos[..., 1]), sin[..., 1],
        torch.zeros_like(cos[..., 1]), torch.ones_like(cos[..., 1]), torch.zeros_like(cos[..., 1]),
        -sin[..., 1], torch.zeros_like(cos[..., 1]), cos[..., 1]
    ], dim=-1).reshape(-1, 3, 3)  # Shape: (N * K, 3, 3)

    # Create rotation matrices around Z-axis
    R_z = torch.stack([
        cos[..., 2], -sin[..., 2], torch.zeros_like(cos[..., 2]),
        sin[..., 2], cos[..., 2], torch.zeros_like(cos[..., 2]),
        torch.zeros_like(cos[..., 2]), torch.zeros_like(cos[..., 2]), torch.ones_like(cos[..., 2])
    ], dim=-1).reshape(-1, 3, 3)  # Shape: (N * K, 3, 3)

    # Combine rotation matrices: R = Rz * Ry * Rx
    R_matrix = torch.bmm(R_z, torch.bmm(R_y, R_x))  # Shape: (N * K, 3, 3)

    # Reshape diff for batch matrix multiplication
    diff = diff.view(-1, 3, 1)  # Shape: (N * K, 3, 1)

    # Apply rotation: R * diff
    diff_rotated = torch.bmm(R_matrix, diff).view(N, K, 3)  # Shape: (N, K, 3)

    # Normalize by sizes
    norm_diff = diff_rotated / sizes_expanded  # Shape: (N, K, 3)
    abs_diff = torch.abs(norm_diff) + 1e-6     # Shape: (N, K, 3)

    # Compute SDF based on superquadric implicit function
    # Adjusted to handle exponents approaching zero
    epsilon1 = exponents_expanded[..., 0]
    epsilon2 = exponents_expanded[..., 1]
    # Avoid division by zero
    epsilon1 = torch.clamp(epsilon1, min=1e-4)
    epsilon2 = torch.clamp(epsilon2, min=1e-4)

    term1 = (abs_diff[..., 0] ** (2 / epsilon2)) + \
            (abs_diff[..., 1] ** (2 / epsilon2))
    term2 = (term1 ** (epsilon2 / epsilon1)) + \
            (abs_diff[..., 2] ** (2 / epsilon1))

    inside = term2 ** (epsilon1 / 2)
    sdf = (inside - 1) * torch.min(sizes_expanded, dim=-1)[0]

    return sdf  # Shape: (N, K)


class Decoder(nn.Module):
    """
    Decoder network that maps encoded features to superquadric parameters.

    Each superquadric is described by 11 parameters:
    - 3 for translation (tx, ty, tz)
    - 3 for rotation (Euler angles: rx, ry, rz)
    - 3 for size (alpha1, alpha2, alpha3)
    - 2 for shape (epsilon1, epsilon2)
    """
    def __init__(self, num_superquadrics=12):
        super(Decoder, self).__init__()
        in_ch = 256
        out_ch = num_superquadrics * 11  # 11 parameters per superquadric
        feat_ch = 512

        # First part of the network: feature transformation
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

        # Second part of the network: parameter prediction
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
        """
        Forward pass of the Decoder.

        Args:
            z (torch.Tensor): Encoded features from the encoder. Shape: (batch_size, in_ch)

        Returns:
            torch.Tensor: Predicted superquadric parameters. Shape: (batch_size, num_superquadrics * 11)
        """
        out1 = self.net1(z)                        # Shape: (batch_size, feat_ch - in_ch)
        in2 = torch.cat([out1, z], dim=-1)         # Concatenate transformed features with original features
        out2 = self.net2(in2)                       # Shape: (batch_size, num_superquadrics * 11)
        return out2


class SQNet(nn.Module):
    """
    SQNet model that encodes surface points and decodes them into multiple superquadrics.

    The network consists of:
    - An encoder (DGCNNFeat) that processes surface point clouds.
    - A decoder that predicts superquadric parameters.
    """
    def __init__(self, num_superquadrics=12):
        super(SQNet, self).__init__()
        self.num_superquadrics = num_superquadrics
        self.encoder = DGCNNFeat(global_feat=True)
        self.decoder = Decoder(num_superquadrics=num_superquadrics)

    def forward(self, surface_points, query_points):
        """
        Forward pass of SQNet.

        Args:
            surface_points (torch.Tensor): Tensor containing surface point clouds. Shape: (batch_size, channels, num_points)
            query_points (torch.Tensor): Tensor containing query points for SDF computation. Shape: (N, 3)

        Returns:
            tuple:
                - torch.Tensor: SDF values for each query point with respect to each superquadric. Shape: (N, K)
                - torch.Tensor: Superquadric parameters. Shape: (batch_size, K, 11)
        """
        # Encode surface points to obtain feature vectors
        features = self.encoder(surface_points)  # Shape: (batch_size, in_ch)

        # Decode features to predict superquadric parameters
        superquadric_params = self.decoder(features)  # Shape: (batch_size, num_superquadrics * 11)

        # Reshape parameters to (batch_size, num_superquadrics, 11)
        superquadric_params = torch.sigmoid(superquadric_params.view(-1, self.num_superquadrics, 11))

        # Define scaling and shifting for each parameter
        # Translation: Scale to [-0.5, 0.5] for each axis
        translation_shift = torch.tensor([-0.5, -0.5, -0.5], device=superquadric_params.device).view(1, 1, 3)
        translation_scale = torch.tensor([1.0, 1.0, 1.0], device=superquadric_params.device).view(1, 1, 3)

        # Rotation: Euler angles scaled to [-pi, pi]
        rotation_shift = torch.tensor([-np.pi, -np.pi, -np.pi], device=superquadric_params.device).view(1, 1, 3)
        rotation_scale = torch.tensor([2 * np.pi, 2 * np.pi, 2 * np.pi], device=superquadric_params.device).view(1, 1, 3)

        # Size: Scale to [0.1, 0.5] for each axis
        size_shift = torch.tensor([0.1, 0.1, 0.1], device=superquadric_params.device).view(1, 1, 3)
        size_scale = torch.tensor([0.4, 0.4, 0.4], device=superquadric_params.device).view(1, 1, 3)

        # Shape: Scale to [0.01, 1.0] for each exponent (allowing sharper edges)
        shape_shift = torch.tensor([0.01, 0.01], device=superquadric_params.device).view(1, 1, 2)
        shape_scale = torch.tensor([0.99, 0.99], device=superquadric_params.device).view(1, 1, 2)

        # Apply scaling and shifting to obtain final parameters
        translations = superquadric_params[:, :, 0:3] * translation_scale + translation_shift  # Shape: (batch_size, K, 3)
        rotations = superquadric_params[:, :, 3:6] * rotation_scale + rotation_shift        # Shape: (batch_size, K, 3)
        sizes = superquadric_params[:, :, 6:9] * size_scale + size_shift                  # Shape: (batch_size, K, 3)
        shapes = superquadric_params[:, :, 9:11] * shape_scale + shape_shift              # Shape: (batch_size, K, 2)

        # Concatenate all parameters to form the final superquadric parameters
        # Final superquadric_params shape: (batch_size, K, 11) = translations (3) + rotations (3) + sizes (3) + shapes (2)
        superquadric_params = torch.cat([translations, rotations, sizes, shapes], dim=2)

        # Compute the Signed Distance Function (SDF) for the query points
        superquadric_sdf = determine_superquadric_sdf(query_points, superquadric_params)  # Shape: (N, K)

        return superquadric_sdf, superquadric_params


def visualise_superquadrics(superquadric_params, reference_model, save_path=None):
    """
    Visualize superquadrics based on their parameters and optionally save the visualization.

    Args:
        superquadric_params (torch.Tensor): Tensor of superquadric parameters. Shape: (batch_size, K, 11)
        reference_model (trimesh.Trimesh or trimesh.PointCloud, optional): Reference 3D model for comparison.
        save_path (str, optional): Path to save the visualization as an OBJ file. Defaults to None.
    """
    # Convert parameters to NumPy for visualization
    superquadric_params = superquadric_params.cpu().detach().numpy()
    translations = superquadric_params[..., :3]   # Shape: (batch_size, K, 3)
    rotations = superquadric_params[..., 3:6]     # Shape: (batch_size, K, 3) - Euler angles
    sizes = superquadric_params[..., 6:9]        # Shape: (batch_size, K, 3)
    exponents = superquadric_params[..., 9:11]   # Shape: (batch_size, K, 2)

    # Initialize a Trimesh scene
    scene = trimesh.Scene()

    # Add reference model to the scene if provided
    if reference_model is not None:
        if isinstance(reference_model, trimesh.points.PointCloud):
            # Assign a blue color to the reference point cloud
            reference_model.colors = [[0, 0, 255, 255]] * len(reference_model.vertices)
        scene.add_geometry(reference_model)

    # Assuming batch_size = 1
    translations = translations[0]  # Shape: (K, 3)
    rotations = rotations[0]        # Shape: (K, 3)
    sizes = sizes[0]                # Shape: (K, 3)
    exponents = exponents[0]        # Shape: (K, 2)

    # Iterate over each superquadric to create and add its mesh to the scene
    for i, (translation, rotation, size, exponent) in enumerate(zip(translations, rotations, sizes, exponents)):
        # Generate the superquadric mesh
        superquadric_mesh = create_superquadric_mesh(size, exponent, rotation, translation)
        # Assign a random color to each superquadric for better visualization
        color = np.random.randint(0, 255, size=3).tolist() + [255]  # RGB + Alpha
        superquadric_mesh.visual.face_colors = color
        # Add the transformed superquadric mesh to the scene
        scene.add_geometry(superquadric_mesh)

    # Save the visualization to a file if a save path is provided
    if save_path is not None:
        scene.export(save_path)

    # Display the scene
    scene.show()


def create_superquadric_mesh(size, exponent, rotation, translation, num_u=50, num_v=50):
    """
    Create a superquadric mesh given the parameters.

    Args:
        size (array-like): Size parameters [a1, a2, a3].
        exponent (array-like): Shape exponents [epsilon1, epsilon2].
        rotation (array-like): Euler angles for rotation [rx, ry, rz].
        translation (array-like): Translation vector [tx, ty, tz].
        num_u (int): Number of samples in the u-direction.
        num_v (int): Number of samples in the v-direction.

    Returns:
        trimesh.Trimesh: The mesh of the superquadric.
    """
    a1, a2, a3 = size
    epsilon1, epsilon2 = exponent

    # Create a grid in parameter space
    u = np.linspace(-np.pi / 2, np.pi / 2, num_u)
    v = np.linspace(-np.pi, np.pi, num_v)
    u, v = np.meshgrid(u, v)

    # Compute the superquadric surface
    cos_u = np.cos(u)
    cos_v = np.cos(v)
    sin_u = np.sin(u)
    sin_v = np.sin(v)

    # Compute the superquadric functions
    fu = np.sign(cos_u) * (np.abs(cos_u) ** epsilon1)
    fv = np.sign(cos_v) * (np.abs(cos_v) ** epsilon2)
    gu = np.sign(sin_u) * (np.abs(sin_u) ** epsilon1)
    gv = np.sign(sin_v) * (np.abs(sin_v) ** epsilon2)

    # Compute coordinates
    x = a1 * fu * fv
    y = a2 * gu * fv
    z = a3 * gu * gv

    # Flatten the arrays
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    # Stack into vertices
    vertices = np.vstack((x, y, z)).T

    # Create faces
    faces = []
    for i in range(num_u - 1):
        for j in range(num_v - 1):
            idx = i * num_v + j
            faces.append([idx, idx + num_v, idx + num_v + 1])
            faces.append([idx, idx + num_v + 1, idx + 1])

    # Convert to numpy array
    faces = np.array(faces)

    # Create the mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Generate rotation matrix from Euler angles
    rot = R.from_euler('xyz', rotation)
    rotation_matrix = rot.as_matrix()  # Shape: (3, 3)

    # Create a 4x4 transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation

    # Apply the 4x4 transformation matrix to the mesh
    mesh.apply_transform(transform_matrix)

    return mesh


def compute_overlap_loss(superquadric_params):
    """
    Compute the overlap loss to minimize overlapping between superquadrics.

    Args:
        superquadric_params (torch.Tensor): Superquadric parameters. Shape: (batch_size, K, 11)

    Returns:
        torch.Tensor: Scalar tensor representing the overlap loss.
    """
    translations = superquadric_params[:, :, :3]  # Shape: (batch_size, K, 3)
    sizes = superquadric_params[:, :, 6:9]        # Shape: (batch_size, K, 3)

    batch_size, K, _ = translations.shape
    overlap_loss = 0.0

    for b in range(batch_size):
        for i in range(K):
            for j in range(i + 1, K):
                center_distance = torch.norm(translations[b, i] - translations[b, j])
                size_i = torch.max(sizes[b, i])
                size_j = torch.max(sizes[b, j])
                min_distance = size_i + size_j
                # If centers are closer than the sum of sizes, there's an overlap
                overlap = F.relu(min_distance - center_distance)
                overlap_loss += overlap

    # Normalize the overlap loss
    overlap_loss = overlap_loss / (batch_size * K * (K - 1) / 2)
    return overlap_loss

def main():
    """
    Main function to train the SQNet model on multiple objects and visualize the resulting superquadrics.
    """
    dataset_root_path = "./reference_models_processed"  # Root directory for the dataset
    object_names = ["dog"]  # List of object names to process
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
    num_epochs = 400  # Number of training epochs

    for obj_name in object_names:
        print(f"Processing object: {obj_name}")
        
        object_path = os.path.join(dataset_root_path, obj_name)
        try:
            # Load the mesh model and surface point cloud using trimesh
            mesh_model = trimesh.load(os.path.join(object_path, f"{obj_name}.obj"))
            if isinstance(mesh_model, list):
                print(f"{obj_name}.obj contains multiple meshes. Selecting the first one.")
                mesh_model = mesh_model[0]  # Select the first mesh
            
            pcd_model = trimesh.load(os.path.join(object_path, "surface_points.ply"))
            if isinstance(pcd_model, list):
                print(f"surface_points.ply for {obj_name} contains multiple point clouds. Selecting the first one.")
                pcd_model = pcd_model[0]  # Select the first point cloud
            
            # Load query points and their corresponding SDF values from a NumPy file
            data_npz = np.load(os.path.join(object_path, "voxel_and_sdf.npz"))
            points, values = data_npz["sdf_points"], data_npz["sdf_values"]
            
            # Convert the loaded data to PyTorch tensors and move them to the specified device
            points = torch.from_numpy(points).float().to(device)            # Shape: [N, 3] or [1, N, 3]
            values = torch.from_numpy(values).float().to(device)            # Shape: [N,]
            surface_pointcloud = torch.from_numpy(pcd_model.vertices).float().to(device)  # Shape: [num_points, 3]
            
            # Check and adjust the shape of points
            print(f"Original points shape: {points.shape}")
            if points.dim() == 3 and points.size(0) == 1:
                points = points.squeeze(0)  # Remove the first dimension
                print(f"Adjusted points shape after squeezing: {points.shape}")
            else:
                print(f"Points shape is already correct: {points.shape}")
            
            # Initialize the SQNet model and move it to the specified device
            model = SQNet(num_superquadrics=12).to(device)

            # Initialize the Adam optimizer with a learning rate of 0.0005
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

            # Training loop
            for epoch in range(num_epochs):
                optimizer.zero_grad()  # Reset gradients

                # Forward pass: compute SDF values and superquadric parameters
                superquadric_sdf, superquadric_params = model(
                    surface_pointcloud.unsqueeze(0).transpose(2, 1), points
                )  # surface_pointcloud shape: [1, 3, num_points], points shape: [N, 3]

                # Combine multiple SDFs using a smooth minimum function along the superquadrics dimension
                superquadric_sdf_combined = bsmin(superquadric_sdf, dim=-1).to(device)  # Shape: [N,]

                # Compute Mean Squared Error (MSE) loss between predicted SDFs and ground truth values
                sdf_loss = torch.mean((superquadric_sdf_combined - values) ** 2)

                # Shape regularization loss
                shape_regularization = torch.mean(1.0 / (superquadric_params[:, :, 9:11] + 1e-6))

                # Overlap loss
                overlap_loss = compute_overlap_loss(superquadric_params)

                # Sparsity loss
                sparsity_loss = torch.mean(torch.exp(-torch.sum(superquadric_params[:, :, 6:9], dim=2)))

                # Total loss with weighted additional losses
                loss = sdf_loss + 0.1 * shape_regularization + 0.1 * overlap_loss + 0.1 * sparsity_loss

                # Backward pass: compute gradients
                loss.backward()

                # Update model parameters
                optimizer.step()

                # Print loss every 50 epochs
                if epoch % 50 == 0:
                    print(f"[{obj_name}] Epoch {epoch}, Loss: {loss.item()}, "
                          f"SDF Loss: {sdf_loss.item()}, Shape Reg: {shape_regularization.item()}, "
                          f"Overlap Loss: {overlap_loss.item()}, Sparsity Loss: {sparsity_loss.item()}")

            # Save the learned superquadric parameters to a NumPy file
            output_dir = os.path.join("./output", obj_name)
            os.makedirs(output_dir, exist_ok=True)
            superquadric_params_np = superquadric_params.cpu().detach().numpy()  # Shape: [1, K, 11]
            np.save(os.path.join(output_dir, f"{obj_name}_superquadric_params.npy"), superquadric_params_np)

            # Visualize the superquadrics and optionally save the visualization
            visualise_superquadrics(
                superquadric_params, reference_model=pcd_model,
                save_path=os.path.join(output_dir, f"{obj_name}_superquadrics.obj")
            )

            print(f"Finished processing {obj_name}. Results saved in {output_dir}.")

        except Exception as e:
            print(f"Error processing {obj_name}: {e}")
            traceback.print_exc()  # Print full traceback



if __name__ == "__main__":
    main()
