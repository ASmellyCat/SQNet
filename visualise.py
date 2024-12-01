import torch
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
import argparse

def visualise_superquadrics_from_np(save_path, reference_model_path=None, output_obj_path=None):
    """
    Visualize superquadrics from saved NumPy parameters.

    Args:
        save_path (str): Path to the saved superquadric_params.npy file.
        reference_model_path (str, optional): Path to the reference 3D model for comparison. Defaults to None.
        output_obj_path (str, optional): Path to save the visualization as an OBJ file. Defaults to None.
    """
    # Load the saved superquadric parameters
    superquadric_params_np = np.load(save_path)  # Shape: [1, K, 11]

    # Extract parameters
    translations = superquadric_params_np[..., :3]   # Shape: (1, K, 3)
    rotations = superquadric_params_np[..., 3:6]     # Shape: (1, K, 3) - Euler angles
    sizes = superquadric_params_np[..., 6:9]        # Shape: (1, K, 3)
    exponents = superquadric_params_np[..., 9:11]   # Shape: (1, K, 2)

    # Initialize a Trimesh scene
    scene = trimesh.Scene()

    # Load and add reference model if provided
    if reference_model_path is not None:
        reference_model = trimesh.load(reference_model_path)
        if isinstance(reference_model, trimesh.points.PointCloud):
            # Assign a blue color to the reference point cloud
            reference_model.colors = [[0, 0, 255, 255]] * len(reference_model.vertices)
        scene.add_geometry(reference_model)

    # Extract the first batch
    translations = translations[0]  # Shape: (K, 3)
    rotations = rotations[0]        # Shape: (K, 3)
    sizes = sizes[0]                # Shape: (K, 3)

    # Iterate over each superquadric
    for i, (translation, rotation, size) in enumerate(zip(translations, rotations, sizes)):
        # Create an icosphere as an approximation of the superquadric surface
        superquadric = trimesh.creation.icosphere(subdivisions=3, radius=1.0)

        # Scale the icosphere based on the superquadric's size parameters
        superquadric.apply_scale(size)  # size: [3]

        # Generate rotation matrix from Euler angles
        rot = R.from_euler('xyz', rotation)  # Assuming 'xyz' rotation order
        rotation_matrix = rot.as_matrix()

        # Create a 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix

        # Apply rotation to the superquadric mesh
        superquadric.apply_transform(transform_matrix)

        # Translate the superquadric mesh to its position in the scene
        superquadric.apply_translation(translation)

        # Add the transformed superquadric mesh to the scene
        scene.add_geometry(superquadric)

    # Save the visualization to a file if a save path is provided
    if output_obj_path is not None:
        scene.export(output_obj_path)

    # Display the scene
    scene.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Superquadrics from Saved Parameters")
    parser.add_argument("--object_name", type=str, required=True, help="Name of the object (e.g., dog, sofa, rod)")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory where output is saved")
    parser.add_argument("--reference_dir", type=str, default="./reference_models_processed", help="Directory with reference models")

    args = parser.parse_args()

    # 构造路径
    object_name = args.object_name
    superquadric_params_path = f"{args.output_dir}/{object_name}/{object_name}_superquadric_params.npy"
    reference_model_path = f"{args.reference_dir}/{object_name}/surface_points.ply"  # 可选
    output_obj_path = f"{args.output_dir}/{object_name}/{object_name}_superquadrics_visualization.obj"  # 可选

    # 调用可视化函数
    visualise_superquadrics_from_np(superquadric_params_path, reference_model_path, output_obj_path)
