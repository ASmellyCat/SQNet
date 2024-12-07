import trimesh
import numpy as np
import numpy as np
import trimesh

def calculate_chamfer_distance(gt, pr):
    """
    Compute the Chamfer Distance between two point clouds using trimesh.
    Args:
        gt (np.ndarray): Nx3 array of ground truth points.
        pr (np.ndarray): Mx3 array of predicted points.
    Returns:
        float: Chamfer Distance.
    """
    gt_cloud = trimesh.points.PointCloud(gt)
    pr_cloud = trimesh.points.PointCloud(pr)
    
    d1 = gt_cloud.kdtree.query(pr_cloud.vertices, k=1)[0]
    d2 = pr_cloud.kdtree.query(gt_cloud.vertices, k=1)[0]
    
    chamfer_dist = np.mean(d1) + np.mean(d2)
    return chamfer_dist

def calculate_fscore(gt, pr, th=0.05):
    """
    Compute the distance between two point clouds using trimesh.
    Args:
        points1 (np.ndarray): Nx3 array of points.
        points2 (np.ndarray): Mx3 array of points.
    Returns:
        np.ndarray: Distances from each point in points1 to the nearest point in points2.
    """
    gt_cloud = trimesh.points.PointCloud(gt)
    pr_cloud = trimesh.points.PointCloud(pr)
    
    d1 = gt_cloud.kdtree.query(pr_cloud.vertices, k=1)[0]
    d2 = pr_cloud.kdtree.query(gt_cloud.vertices, k=1)[0]

    recall = float(sum(d < th for d in d2)) / float(len(d2))
    precision = float(sum(d < th for d in d1)) / float(len(d1))
    if recall+precision > 0:
        fscore = 2 * recall * precision / (recall + precision)
    else:
        fscore = 0
    return fscore, precision, recall

def calculate_metrics(gt_point_cloud, cuboid_params, num_surface_points):
    # Sample points from the cuboid surfaces
    sampled_points = []
    for i in range(cuboid_params.shape[0]):
        center = cuboid_params[i, :3]
        quaternion = cuboid_params[i, 3:7]
        dimensions = cuboid_params[i, 7:]

        box = trimesh.creation.box(extents=dimensions.cpu().detach().numpy())
        transform = trimesh.transformations.quaternion_matrix(quaternion.cpu().detach().numpy())
        transform[:3, 3] = center.cpu().detach().numpy()
        box.apply_transform(transform)

        sampled_points.append(box.sample(num_surface_points))

    sampled_points = np.concatenate(sampled_points, axis=0)

    # Compute F1-score    
    fscore, precision, recall = calculate_fscore(gt_point_cloud, sampled_points)
    print(f"F1-score (↑): {fscore:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}")
    # Compute Chamfer Distance
    chamfer_distance = calculate_chamfer_distance(gt_point_cloud, sampled_points)
    print(f"Chamfer Distance (↓): {chamfer_distance:.6f}")