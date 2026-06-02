# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import time
from pathlib import Path

import warp as wp

import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.assets.csv as csv_utils
import soma_retargeter.pipelines.newton_pipeline as newton_pipeline
from soma_retargeter.utils.space_conversion_utils import (
    SpaceConverter,
    get_facing_direction_type_from_str,
)


def _elapsed(seconds: float) -> str:
    return f"{int(seconds // 3600):02d}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}"


def _append_row(path: Path, fieldnames: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _load_batch(paths: list[Path], skeleton) -> tuple[list[Path], list]:
    loaded_paths = []
    animations = []
    for path in paths:
        _, animation = bvh_utils.load_bvh(path, skeleton)
        loaded_paths.append(path)
        animations.append(animation)
    return loaded_paths, animations


def _run_batch(
    pipeline,
    paths: list[Path],
    animations: list,
    source_xform,
    import_root: Path,
    export_root: Path,
    csv_config,
) -> int:
    pipeline.clear()
    pipeline.add_input_motions(animations, [source_xform] * len(animations), True)
    buffers = pipeline.execute()
    for path, buffer in zip(paths, buffers):
        dst_path = export_root / path.relative_to(import_root).with_suffix(".csv")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        csv_utils.save_csv(str(dst_path), buffer, csv_config=csv_config)
    return len(buffers)


def _retarget_with_split(
    pipeline,
    paths: list[Path],
    animations: list,
    source_xform,
    import_root: Path,
    export_root: Path,
    csv_config,
    failures_path: Path,
) -> int:
    try:
        return _run_batch(pipeline, paths, animations, source_xform, import_root, export_root, csv_config)
    except Exception as exc:
        if len(paths) == 1:
            _append_row(
                failures_path,
                ["path", "error"],
                {"path": str(paths[0]), "error": repr(exc)},
            )
            print(f"[WARN] failed {paths[0]}: {exc!r}", flush=True)
            return 0
        mid = len(paths) // 2
        print(f"[WARN] batch failed ({len(paths)} clips), splitting: {exc!r}", flush=True)
        done = _retarget_with_split(
            pipeline,
            paths[:mid],
            animations[:mid],
            source_xform,
            import_root,
            export_root,
            csv_config,
            failures_path,
        )
        done += _retarget_with_split(
            pipeline,
            paths[mid:],
            animations[mid:],
            source_xform,
            import_root,
            export_root,
            csv_config,
            failures_path,
        )
        return done


def _collect_manifest_paths(
    import_root: Path,
    manifest_csv: Path,
    manifest_path_column: str,
) -> tuple[list[Path], list[str]]:
    paths = []
    missing = []
    with manifest_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or manifest_path_column not in reader.fieldnames:
            raise ValueError(
                f"Manifest {manifest_csv} does not contain column {manifest_path_column!r}; "
                f"columns={reader.fieldnames}"
            )
        for row in reader:
            relpath = row[manifest_path_column].strip()
            path = (import_root / relpath).resolve()
            if path.exists():
                paths.append(path)
            else:
                missing.append(relpath)
    return paths, missing


def _collect_paths(
    import_root: Path,
    export_root: Path,
    resume: bool,
    limit: int | None,
    manifest_csv: Path | None,
    manifest_path_column: str,
    num_shards: int,
    shard_index: int,
) -> tuple[list[Path], int, list[str], int]:
    if manifest_csv is None:
        all_paths = list(import_root.rglob("*.bvh"))
        missing = []
    else:
        all_paths, missing = _collect_manifest_paths(import_root, manifest_csv.resolve(), manifest_path_column)

    # Sort by size before round-robin sharding so long clips are balanced across workers.
    all_paths = sorted(all_paths, key=lambda p: p.stat().st_size, reverse=True)
    if limit is not None:
        all_paths = all_paths[:limit]

    pre_shard_count = len(all_paths)
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < --num-shards")
    all_paths = all_paths[shard_index::num_shards]

    if not resume:
        return all_paths, 0, missing, pre_shard_count

    pending = []
    skipped = 0
    for path in all_paths:
        dst_path = export_root / path.relative_to(import_root).with_suffix(".csv")
        if dst_path.exists() and dst_path.stat().st_size > 0:
            skipped += 1
        else:
            pending.append(path)
    return pending, skipped, missing, pre_shard_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume-friendly batch BVH to robot CSV exporter.")
    parser.add_argument("--import-root", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--robot-type", default="unitree_h2_sonic")
    parser.add_argument("--retarget-source", default="soma")
    parser.add_argument("--source-facing-direction", default="Mujoco")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--manifest-path-column", default="soma_relpath")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress", type=Path, default=None)
    parser.add_argument("--failures", type=Path, default=None)
    args = parser.parse_args()

    import_root = args.import_root.resolve()
    export_root = args.export_root.resolve()
    export_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.progress or (export_root.parent / "logs" / "batch_progress.csv")
    failures_path = args.failures or (export_root.parent / "logs" / "batch_failures.csv")

    paths, skipped, missing, pre_shard_count = _collect_paths(
        import_root,
        export_root,
        args.resume,
        args.limit,
        args.manifest_csv,
        args.manifest_path_column,
        args.num_shards,
        args.shard_index,
    )
    total_pending = len(paths)
    print(f"[INFO] import_root={import_root}", flush=True)
    print(f"[INFO] export_root={export_root}", flush=True)
    if args.manifest_csv is not None:
        print(
            f"[INFO] manifest={args.manifest_csv.resolve()} column={args.manifest_path_column} "
            f"manifest_existing={pre_shard_count} missing={len(missing)}",
            flush=True,
        )
    print(
        f"[INFO] shard={args.shard_index}/{args.num_shards} pending={total_pending} "
        f"skipped_existing={skipped} batch_size={args.batch_size}",
        flush=True,
    )
    for relpath in missing:
        _append_row(failures_path, ["path", "error"], {"path": relpath, "error": "missing_manifest_path"})
    if not paths:
        print("[OK] no pending BVH files", flush=True)
        return

    bvh_importer = bvh_utils.BVHImporter()
    skeleton, _ = bvh_importer.create_skeleton(paths[0])
    converter = SpaceConverter(get_facing_direction_type_from_str(args.source_facing_direction))
    source_xform = converter.transform(wp.transform_identity())
    csv_config = csv_utils.get_csv_config(args.robot_type)

    completed = 0
    failed_load = 0
    started = time.time()
    with wp.ScopedDevice(args.device):
        pipeline = newton_pipeline.NewtonPipeline(skeleton, args.retarget_source, args.robot_type)
        for batch_idx, first in enumerate(range(0, len(paths), args.batch_size), start=1):
            batch_paths = paths[first : first + args.batch_size]
            batch_start = time.time()
            print(
                f"[INFO] batch={batch_idx} clips={len(batch_paths)} "
                f"completed={completed}/{total_pending}",
                flush=True,
            )
            try:
                loaded_paths, animations = _load_batch(batch_paths, skeleton)
            except Exception as exc:
                if len(batch_paths) == 1:
                    failed_load += 1
                    _append_row(failures_path, ["path", "error"], {"path": str(batch_paths[0]), "error": repr(exc)})
                    continue
                for path in batch_paths:
                    try:
                        one_paths, one_anims = _load_batch([path], skeleton)
                    except Exception as one_exc:
                        failed_load += 1
                        _append_row(failures_path, ["path", "error"], {"path": str(path), "error": repr(one_exc)})
                        print(f"[WARN] failed to load {path}: {one_exc!r}", flush=True)
                        continue
                    completed += _retarget_with_split(
                        pipeline,
                        one_paths,
                        one_anims,
                        source_xform,
                        import_root,
                        export_root,
                        csv_config,
                        failures_path,
                    )
                continue

            completed += _retarget_with_split(
                pipeline,
                loaded_paths,
                animations,
                source_xform,
                import_root,
                export_root,
                csv_config,
                failures_path,
            )
            elapsed = time.time() - started
            batch_elapsed = time.time() - batch_start
            _append_row(
                progress_path,
                [
                    "batch",
                    "shard_index",
                    "num_shards",
                    "completed",
                    "pending_total",
                    "skipped_existing",
                    "failed_load",
                    "batch_elapsed_s",
                    "elapsed_s",
                    "clips_per_hour",
                ],
                {
                    "batch": batch_idx,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "completed": completed,
                    "pending_total": total_pending,
                    "skipped_existing": skipped,
                    "failed_load": failed_load,
                    "batch_elapsed_s": f"{batch_elapsed:.3f}",
                    "elapsed_s": f"{elapsed:.3f}",
                    "clips_per_hour": f"{(completed / max(elapsed, 1e-6)) * 3600.0:.3f}",
                },
            )
            print(
                f"[INFO] batch={batch_idx} done completed={completed}/{total_pending} "
                f"elapsed={_elapsed(elapsed)} rate={(completed / max(elapsed, 1e-6)) * 3600.0:.1f}/h",
                flush=True,
            )

    elapsed = time.time() - started
    print(
        f"[OK] completed={completed}/{total_pending} skipped_existing={skipped} "
        f"failed_load={failed_load} elapsed={_elapsed(elapsed)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
