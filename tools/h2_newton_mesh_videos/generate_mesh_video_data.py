# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

import newton
import numpy as np
import warp as wp

import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.pipelines.utils as pipeline_utils
import soma_retargeter.robotics.robot_model as robot_model
import soma_retargeter.utils.newton_utils as newton_utils
from soma_retargeter.animation.skeleton import SkeletonInstance
from soma_retargeter.renderers.mesh_renderer import (
    SkeletalMeshRenderer,
    skinning_kernel,
    update_skinned_transform_kernel,
)
from soma_retargeter.utils.space_conversion_utils import (
    SpaceConverter,
    get_facing_direction_type_from_str,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/h2_newton_videos/data"
DEFAULT_MOTIONS = [
    REPO_ROOT / "assets/motions/bvh/Neutral_walk_forward_002__A057.bvh",
    REPO_ROOT / "assets/motions/bvh/high_jump_R_001__A277.bvh",
    REPO_ROOT / "assets/motions/bvh/dance_hiphop_shuffle_square_R_fast_002__A318.bvh",
    REPO_ROOT / "assets/motions/bvh/big_light_one_hand_pick_up_front_low_R_005__A508.bvh",
    REPO_ROOT / "assets/motions/bvh/wave_R_001__A428.bvh",
]


def _skin_source_mesh(renderer, skeleton_instance):
    animation_transforms = skeleton_instance.get_local_transforms()
    wp.launch(
        update_skinned_transform_kernel,
        dim=1,
        inputs=[
            skeleton_instance.skeleton.num_joints,
            wp.array(animation_transforms, dtype=wp.transform),
            wp.array(skeleton_instance.parent_indices, dtype=wp.int32),
            renderer.skeletal_mesh.bind_transforms,
            skeleton_instance.xform,
        ],
        outputs=[renderer.skinned_transforms],
    )

    vertices = []
    for i, skinned_mesh in enumerate(renderer.skeletal_mesh.skinned_meshes):
        if skinned_mesh.num_points == 0:
            vertices.append(np.zeros((0, 3), dtype=np.float32))
            continue

        wp.launch(
            skinning_kernel,
            dim=skinned_mesh.num_points,
            inputs=[
                skinned_mesh.points,
                skinned_mesh.joint_indices,
                skinned_mesh.joint_weights,
                int(skinned_mesh.num_influences),
                wp.array(renderer.skinned_transforms[0], dtype=wp.transform),
            ],
            outputs=[renderer.skinned_points[i]],
        )
        vertices.append(renderer.skinned_points[i].numpy().astype(np.float32, copy=True))

    return vertices


def _make_h2_model():
    target_builder = robot_model.create_robot_builder("unitree_h2")
    builder = newton.ModelBuilder()
    builder.add_builder(target_builder, wp.transform_identity())
    model = builder.finalize()
    state = model.state()
    body_names = [newton_utils.get_name_from_label(label) for label in target_builder.body_label]
    return model, state, body_names


def _retarget_motion(skeleton, animation):
    import soma_retargeter.pipelines.newton_pipeline as newton_pipeline

    pipeline = newton_pipeline.NewtonPipeline(skeleton, "soma", "unitree_h2")
    converter = SpaceConverter(get_facing_direction_type_from_str("Mujoco"))
    source_xform = converter.transform(wp.transform_identity())
    source_rot = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(*source_xform[3:7]))
    pipeline.add_input_motions([animation], [source_rot], True)
    return pipeline.execute()[0], source_xform


def _sample_motion(motion_path, out_path, fps, max_seconds=None):
    skeleton, animation = bvh_utils.load_bvh(str(motion_path))
    csv_buffer, source_xform = _retarget_motion(skeleton, animation)

    total_time = animation.num_frames / animation.sample_rate
    if max_seconds is not None:
        total_time = min(total_time, max_seconds)
    sample_count = max(2, int(np.ceil(total_time * fps)))
    times = np.arange(sample_count, dtype=np.float32) / float(fps)

    robot_model_state, state, body_names = _make_h2_model()
    robot_offset = wp.transform(wp.vec3(0.0, -0.5, 0.0), wp.quat_identity())
    csv_buffer.xform = robot_offset

    source_offset = wp.transform(wp.vec3(0.0, 0.5, 0.0), wp.quat_identity())
    skeleton_instance = SkeletonInstance(skeleton, [235.0 / 255.0, 245.0 / 255.0, 112.0 / 255.0], wp.mul(source_offset, source_xform))
    skeletal_mesh = pipeline_utils.get_source_model_mesh(pipeline_utils.SourceType.SOMA, skeleton)
    mesh_renderer = SkeletalMeshRenderer(skeletal_mesh)

    h2_body_q = np.empty((sample_count, len(body_names), 7), dtype=np.float32)
    soma_vertices = [
        np.empty((sample_count, mesh.num_points, 3), dtype=np.float32)
        for mesh in skeletal_mesh.skinned_meshes
    ]

    for i, t in enumerate(times):
        robot_q = csv_buffer.sample(float(t)).astype(np.float32, copy=False)
        wp.copy(
            robot_model_state.joint_q,
            wp.array(robot_q, dtype=wp.float32),
            0,
            0,
            robot_q.shape[0],
        )
        newton.eval_fk(robot_model_state, robot_model_state.joint_q, robot_model_state.joint_qd, state, None)
        h2_body_q[i] = state.body_q.numpy().astype(np.float32, copy=True)

        skeleton_instance.set_local_transforms(animation.sample(float(t)))
        for mesh_idx, verts in enumerate(_skin_source_mesh(mesh_renderer, skeleton_instance)):
            soma_vertices[mesh_idx][i] = verts

    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "motion": str(motion_path),
        "fps": fps,
        "duration": total_time,
        "body_names": body_names,
        "soma_parts": len(skeletal_mesh.skinned_meshes),
    }
    arrays = {
        "times": times,
        "h2_body_q": h2_body_q,
        "body_names": np.array(body_names),
        "metadata": np.array(json.dumps(metadata)),
    }
    for i, mesh in enumerate(skeletal_mesh.skinned_meshes):
        arrays[f"soma_vertices_{i}"] = soma_vertices[i]
        arrays[f"soma_indices_{i}"] = mesh.indices.numpy().astype(np.int32, copy=True).reshape(-1, 3)
    np.savez_compressed(out_path, **arrays)
    print(f"[OK] wrote {out_path} ({sample_count} frames @ {fps} fps)")


def main():
    parser = argparse.ArgumentParser(
        description="Retarget BVH motions to Unitree H2 and cache mesh poses for Blender video rendering.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("motions", nargs="*", default=[str(path) for path in DEFAULT_MOTIONS])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for motion in args.motions:
        motion_path = Path(motion)
        out_path = out_dir / f"{motion_path.stem}.npz"
        _sample_motion(motion_path, out_path, args.fps, args.max_seconds)


if __name__ == "__main__":
    main()
