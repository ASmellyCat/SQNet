import os
import numpy as np
import torch
import trimesh

DATASET_ROOT_PATH = "./reference_models_processed"

def load_reference_data(obj_name, dataset_root_path=DATASET_ROOT_PATH):
    """
    Load reference mesh and point cloud for the given object name.
    Assign a uniform color to the point cloud.
    """
    object_path = os.path.join(dataset_root_path, obj_name)
    mesh_model_path = os.path.join(object_path, f"{obj_name}.obj")
    pcd_model_path = os.path.join(object_path, "surface_points.ply")
    
    reference_mesh = None
    if os.path.exists(mesh_model_path):
        reference_mesh = trimesh.load(mesh_model_path, force='mesh')
        # Assign a light gray color to the mesh faces
        if hasattr(reference_mesh, 'visual') and hasattr(reference_mesh.visual, 'face_colors'):
            reference_mesh.visual.face_colors = [200, 200, 200, 255]

    reference_pcd = None
    if os.path.exists(pcd_model_path):
        reference_pcd = trimesh.load(pcd_model_path, force='pointcloud')
        if reference_pcd is not None:
            # Light brownish color for point cloud
            reference_pcd.colors = np.array([[219, 204, 188, 255]] * len(reference_pcd.vertices))
            
    return reference_mesh, reference_pcd

def load_cuboid_params(obj_name):
    """
    Load cuboid parameters from npy file.
    """
    param_path = f"goodresult/{obj_name}/{obj_name}_cuboid_params.npy"
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"Cuboid parameters not found at: {param_path}")
    cuboid_params = np.load(param_path)
    return cuboid_params

def create_cuboid_meshes(cuboid_params, use_neon=False):
    """
    Create a list of cuboid meshes from parameters.
    Args:
        cuboid_params: Nx10 array (center(3), quaternion(4), dimensions(3))
        use_neon: Not needed here, just normal colors.
    """
    cuboid_centers = cuboid_params[:, :3]
    cuboid_quaternions = cuboid_params[:, 3:7]
    cuboid_dimensions = cuboid_params[:, 7:]
    
    # Normal colors (no neon needed)
    normal_colors = [
        [179, 0, 30, 255],
        [179, 255, 25, 255],
        [255, 224, 58, 255],
        [128, 170, 255, 255],
        [255, 102, 255, 255],
        [255, 119, 51, 255],
        [196, 77, 255, 255],
        [179, 242, 255, 255],
        [128, 128, 0, 255],
    ]
    
    colors = normal_colors
    
    cuboids = []
    for i, (center, quaternion, dimensions) in enumerate(zip(cuboid_centers, cuboid_quaternions, cuboid_dimensions)):
        box = trimesh.creation.box(extents=dimensions)
        transform = trimesh.transformations.quaternion_matrix(quaternion)
        transform[:3, 3] = center
        box.apply_transform(transform)
        
        color = colors[i % len(colors)]
        box.visual.face_colors = color
        cuboids.append(box)
    return cuboids

def create_scene(geometries):
    """
    Create a trimesh scene and add given geometries to it.
    """
    scene = trimesh.Scene()
    for g in geometries:
        scene.add_geometry(g)
    return scene

def visualize_obj_scenes(obj_name):
    """
    Show three scenes in sequence:
    1) Ground truth mesh (light gray)
    2) Cuboid assembly only
    3) Point cloud + cuboid assembly
    """
    reference_mesh, reference_pcd = load_reference_data(obj_name)
    cuboid_params = load_cuboid_params(obj_name)
    
    if isinstance(cuboid_params, torch.Tensor):
        cuboid_params = cuboid_params.cpu().detach().numpy()
    
    # Cuboids (normal colors)
    cuboids = create_cuboid_meshes(cuboid_params, use_neon=False)
    
    # Scene 1: Just ground truth mesh (if available)
    scene1 = create_scene([])
    if reference_mesh is not None:
        scene1.add_geometry(reference_mesh.copy())
    print("Showing scene 1: Just ground truth mesh.")
    scene1.show()  # Close this window after screenshot
    
    # Scene 2: Just cuboids
    scene2 = create_scene([])
    for c in cuboids:
        scene2.add_geometry(c.copy())
    print("Showing scene 2: Just cuboid assembly.")
    scene2.show()  # Close this window after screenshot
    
    # Scene 3: point cloud + cuboids
    scene3 = create_scene([])
    if reference_pcd is not None:
        scene3.add_geometry(reference_pcd.copy())
    for c in cuboids:
        scene3.add_geometry(c.copy())
    print("Showing scene 3: Point cloud + cuboids.")
    scene3.show()  # Close this window after screenshot

# Call the function
visualize_obj_scenes("hand")
