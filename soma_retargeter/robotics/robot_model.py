# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import newton

import soma_retargeter.utils.io_utils as io_utils


_SUPPORTED_ROBOT_TYPES = ("unitree_g1", "unitree_h2")


def get_supported_robot_types() -> list[str]:
    return list(_SUPPORTED_ROBOT_TYPES)


def get_robot_mjcf_path(robot_type: str) -> Path:
    if robot_type == "unitree_g1":
        return newton.utils.download_asset("unitree_g1") / "mjcf/g1_29dof_rev_1_0.xml"
    if robot_type == "unitree_h2":
        return io_utils.get_package_root() / "robot_assets" / "unitree_h2" / "h2.xml"

    allowed = ", ".join(_SUPPORTED_ROBOT_TYPES)
    raise ValueError(f"Unknown robot type: [{robot_type}]. Allowed values: {allowed}")


def create_robot_builder(robot_type: str) -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    builder.add_mjcf(str(get_robot_mjcf_path(robot_type)))
    return builder
