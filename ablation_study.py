#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ablation Study Script

This script runs four different experiments, each with multiple variations:
1. Dog - Repulsion Weight Variation
2. Hand - Initialization Variation
3. Sofa - Dimension Loss Variation
"""

import os
import torch
import numpy as np
import trimesh

from cuboid_abstraction_withSDF import (
    CuboidNet, SDFNetwork, bsmin, determine_cuboid_sdf, compute_coverage_loss, compute_repulsion_loss,
    compute_consistency_loss, compute_dimension_regularization, initialize_cuboid_params_with_spectral_clustering,
    visualize_cuboids, train_sdf_network, load_trained_sdf_network
)


def run_experiment(
    object_names,
    dataset_root_path,
    output_dir,
    num_epochs,
    num_cuboids,
    learning_rate,
    bsmin_k,
    coverage_weight=0.0,
    rotation_weight=0.0,
    repulsion_weight=0.0,
    dimension_weight=0.0,
    num_surface_points=1000,
    use_sdf_training=False,
    use_init=True,   # If False, skip spectral clustering initialization
    use_sdf_net=True # If False, do not load or use SDF network at all
):
    """
    Run a single ablation experiment with given hyperparameters.

    Args:
        use_init (bool): If True, use spectral clustering initialization; if False, skip it.
        use_sdf_net (bool): If False, do not load or use the SDF network. 
                            If consistency_weight > 0 and use_sdf_net=False, consistency loss will be skipped or must be 0.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for obj_name in object_names:
        print(f"\n[INFO] Running experiment for {obj_name}...")
        obj_output_dir = os.path.join(output_dir, f"{obj_name}_experiment")
        os.makedirs(obj_output_dir, exist_ok=True)

        object_path = os.path.join(dataset_root_path, obj_name)
        mesh_model_path = os.path.join(object_path, f"{obj_name}.obj")
        pcd_model_path = os.path.join(object_path, "surface_points.ply")
        sdf_npz_path = os.path.join(object_path, "voxel_and_sdf.npz")

        # Data loading
        if not os.path.exists(mesh_model_path):
            print(f"[WARNING] Mesh file not found for {obj_name}: {mesh_model_path}")
            continue
        mesh_model = trimesh.load(mesh_model_path)
        if isinstance(mesh_model, list):
            mesh_model = mesh_model[0]

        if not os.path.exists(pcd_model_path):
            print(f"[WARNING] Point cloud not found for {obj_name}: {pcd_model_path}")
            continue
        pcd_model = trimesh.load(pcd_model_path)
        if isinstance(pcd_model, list):
            pcd_model = pcd_model[0]

        if not os.path.exists(sdf_npz_path):
            print(f"[WARNING] SDF data not found for {obj_name}: {sdf_npz_path}")
            continue
        data_npz = np.load(sdf_npz_path)
        points = torch.from_numpy(data_npz["sdf_points"]).float().to(device)
        values = torch.from_numpy(data_npz["sdf_values"]).float().to(device)
        surface_pointcloud = torch.from_numpy(pcd_model.vertices).float().to(device)

        # Model initialization
        model = CuboidNet(num_cuboids=num_cuboids).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Initialization method control
        if use_init:
            initialize_cuboid_params_with_spectral_clustering(model, surface_pointcloud, num_cuboids, device)
        else:
            print("[INFO] Skipping spectral clustering initialization (random init).")

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            cuboid_sdf, cuboid_params = model(
                surface_points=surface_pointcloud.unsqueeze(0).transpose(2, 1),
                query_points=points
            )

            cuboid_sdf_blended = bsmin(cuboid_sdf, dim=-1, k=bsmin_k).to(device)
            mse_loss = torch.mean((cuboid_sdf_blended - values) ** 2)

            quaternions = cuboid_params[:, 3:7]
            rotation_loss_val = torch.mean(torch.abs(quaternions[:, 1:]))

            coverage_loss_val = compute_coverage_loss(cuboid_params, surface_pointcloud) if coverage_weight > 0 else 0.0
            repulsion_loss_val = compute_repulsion_loss(cuboid_params) if repulsion_weight > 0 else 0.0


            dimension_loss_val = compute_dimension_regularization(cuboid_params) if dimension_weight > 0 else 0.0

            loss = (
                mse_loss
                + rotation_weight * rotation_loss_val
                + coverage_weight * coverage_loss_val
                + repulsion_weight * repulsion_loss_val
                + dimension_weight * dimension_loss_val
            )

            loss.backward()
            optimizer.step()

            # Organized printing
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(
                    f"[{obj_name} | Epoch {epoch+1}/{num_epochs}] "
                    f"Total: {loss.item():.6f}, MSE: {mse_loss.item():.6f}, "
                    f"Rot: {rotation_loss_val:.6f}, Cov: {float(coverage_loss_val):.6f} "
                )

        cuboid_params_path = os.path.join(obj_output_dir, f"{obj_name}_cuboid_params.npy")
        np.save(cuboid_params_path, cuboid_params.cpu().detach().numpy())
        print(f"[INFO] Cuboid parameters saved to {cuboid_params_path}")

        cuboids_save_path = os.path.join(obj_output_dir, f"{obj_name}_cuboids.obj")
        visualize_cuboids(cuboid_params=cuboid_params, reference_model=pcd_model, save_path=cuboids_save_path)
        print(f"[INFO] Cuboids visualization saved to {cuboids_save_path}")

    print("[INFO] Experiment finished.\n")


if __name__ == "__main__":
    output_dir = "./ablation_output"
    dataset_root_path = "./reference_models_processed"

    # # ---------------------------
    # # Experiment 1: Dog - Repulsion Weight Variation
    # # ---------------------------
    # print("\n========== Experiment 1: Dog - Repulsion Weight Variation ==========\n")
    # # Baseline
    # print("** Running Baseline (repulsion_weight=0.001) **")
    # run_experiment(
    #     object_names=["dog"],
    #     dataset_root_path=dataset_root_path,
    #     output_dir=output_dir,
    #     num_epochs=1000,
    #     num_cuboids=16,
    #     learning_rate=0.0005,
    #     bsmin_k=22,
    #     coverage_weight=0.1,
    #     rotation_weight=0.01,
    #     repulsion_weight=0.000,  # baseline
    #     dimension_weight=0.0,
    #     num_surface_points=1000,
    #     use_sdf_training=False,
    #     use_init=True
    # )

    # # Variation B1
    # print("** Running Variation B1 (repulsion_weight=0.05) **")
    # run_experiment(
    #     object_names=["dog"],
    #     dataset_root_path=dataset_root_path,
    #     output_dir=output_dir,
    #     num_epochs=1000,
    #     num_cuboids=16,
    #     learning_rate=0.0005,
    #     bsmin_k=22,
    #     coverage_weight=0.1,
    #     rotation_weight=0.01,
    #     repulsion_weight=0.05,
    #     dimension_weight=0.0,
    #     num_surface_points=1000,
    #     use_sdf_training=False,
    #     use_init=True
    # )

    # # Variation B2
    # print("** Running Variation B2 (repulsion_weight=0.1) **")
    # run_experiment(
    #     object_names=["dog"],
    #     dataset_root_path=dataset_root_path,
    #     output_dir=output_dir,
    #     num_epochs=1000,
    #     num_cuboids=16,
    #     learning_rate=0.0005,
    #     bsmin_k=22,
    #     coverage_weight=0.1,
    #     rotation_weight=0.01,
    #     repulsion_weight=0.1,
    #     dimension_weight=0.0,
    #     num_surface_points=1000,
    #     use_sdf_training=False,
    #     use_init=True
    # )

# ---------------------------
    # Experiment 2: Hand - SDF Network On/Off
    # ---------------------------
    print("\n========== Experiment 2: Hand - SDF Network On/Off ==========\n")

    # # Baseline: use_sdf_net=False (no SDF net at all)
    # print("** Running Baseline (use_sdf_net=False) **")
    # run_experiment(
    #     object_names=["hand"],
    #     dataset_root_path=dataset_root_path,
    #     output_dir=output_dir,
    #     num_epochs=1000,
    #     num_cuboids=5,
    #     learning_rate=0.0001,
    #     bsmin_k=22,
    #     coverage_weight=0.0,
    #     rotation_weight=0.0,
    #     repulsion_weight=0.05,
    #     dimension_weight=0.0,
    #     num_surface_points=1000,
    #     use_sdf_training=False,
    #     use_init=True,

    # )

    # # Variation S1: use_sdf_net=True (SDF net available)
    # print("** Running Variation S1 (use_sdf_net=True) **")
    # run_experiment(
    #     object_names=["hand"],
    #     dataset_root_path=dataset_root_path,
    #     output_dir=output_dir,
    #     num_epochs=1000,
    #     num_cuboids=5,
    #     learning_rate=0.0001,
    #     bsmin_k=22,
    #     coverage_weight=0.0,
    #     rotation_weight=0.0,
    #     repulsion_weight=0.05,
    #     dimension_weight=0.0,
    #     num_surface_points=1000,
    #     use_sdf_training=False,
    #     use_init=False,
    # )

    # ---------------------------
    # Experiment 3: Pot - Coverage Loss Variation
    # ---------------------------
    print("\n========== Experiment 3: Pot - Initialization Variation ==========\n")
    # Baseline (With initialization)
    print("** Running Baseline (coverage_weight=0.00) **")
    run_experiment(
        object_names=["pot"],
        dataset_root_path=dataset_root_path,
        output_dir=output_dir,
        num_epochs=1000,
        num_cuboids=6,
        learning_rate=0.0001,
        bsmin_k=22,
        coverage_weight=0.0, #baseline
        rotation_weight=0.05,
        repulsion_weight=0.05,
        dimension_weight=0.0,
        num_surface_points=1000,
        use_sdf_training=False,
        use_init=False  
    )

    # Variation I1 (Without initialization)
    print("** Running Variation I1 (coverage_weight=0.05) **")
    run_experiment(
        object_names=["pot"],
        dataset_root_path=dataset_root_path,
        output_dir=output_dir,
        num_epochs=1000,
        num_cuboids=6,
        learning_rate=0.0001,
        bsmin_k=22,
        coverage_weight=0.05,
        rotation_weight=0.05,
        repulsion_weight=0.05,
        dimension_weight=0.0,
        num_surface_points=1000,
        use_sdf_training=False,
        use_init=False
    )

     # Variation I2 (Without initialization)
    print("** Running Variation I1 (coverage_weight=0.1) **")
    run_experiment(
        object_names=["pot"],
        dataset_root_path=dataset_root_path,
        output_dir=output_dir,
        num_epochs=1000,
        num_cuboids=6,
        learning_rate=0.0001,
        bsmin_k=22,
        coverage_weight=0.1,
        rotation_weight=0.05,
        repulsion_weight=0.05,
        dimension_weight=0.0,
        num_surface_points=1000,
        use_sdf_training=False,
        use_init=False
    )

    # ---------------------------
    # Experiment 3: Sofa - Dimension Loss Variation
    # ---------------------------
    print("\n========== Experiment 4: Sofa - Dimension Loss Variation ==========\n")
    # Baseline (dimension_weight=0.0)
    print("** Running Baseline (dimension_weight=0.0) **")
    run_experiment(
        object_names=["sofa"],
        dataset_root_path=dataset_root_path,
        output_dir=output_dir,
        num_epochs=2000,
        num_cuboids=8,
        learning_rate=0.0005,
        bsmin_k=22,
        coverage_weight=0.3,
        rotation_weight=0.1,
        repulsion_weight=0.001,
        dimension_weight=0.0,  # baseline
        num_surface_points=1000,
        use_sdf_training=False,
        use_init=True
    )

    # Variation D1
    print("** Running Variation D1 (dimension_weight=0.01) **")
    run_experiment(
        object_names=["sofa"],
        dataset_root_path=dataset_root_path,
        output_dir=output_dir,
        num_epochs=2000,
        num_cuboids=8,
        learning_rate=0.0005,
        bsmin_k=22,
        coverage_weight=0.3,
        rotation_weight=0.1,
        repulsion_weight=0.001,
        dimension_weight=0.01,
        num_surface_points=1000,
        use_sdf_training=False,
        use_init=True
    )

    # Variation D2
    print("** Running Variation D2 (dimension_weight=0.05) **")
    run_experiment(
        object_names=["sofa"],
        dataset_root_path=dataset_root_path,
        output_dir=output_dir,
        num_epochs=2000,
        num_cuboids=8,
        learning_rate=0.0005,
        bsmin_k=22,
        coverage_weight=0.3,
        rotation_weight=0.1,
        repulsion_weight=0.001,
        dimension_weight=0.01,
        num_surface_points=1000,
        use_sdf_training=False,
        use_init=True
    )

    print("\n[INFO] All experiments completed successfully!\n")
