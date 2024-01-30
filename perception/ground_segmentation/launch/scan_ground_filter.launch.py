# Copyright 2021 Tier IV, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import DeclareLaunchArgument
from launch.actions import GroupAction
from launch.actions import OpaqueFunction
from launch.conditions import LaunchConfigurationEquals
from launch.conditions import LaunchConfigurationNotEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode
import yaml


def launch_setup(context, *args, **kwargs):
    vehicle_info_param_path = LaunchConfiguration("vehicle_info_param_file").perform(context)
    with open(vehicle_info_param_path, "r") as f:
        vehicle_info_param = yaml.safe_load(f)["/**"]["ros__parameters"]

    # nodes = [
    #     ComposableNode(
    #         package="ground_segmentation",
    #         plugin="ground_segmentation::ScanGroundFilterComponent",
    #         name="scan_ground_filter",
    #         remappings=[
    #             ("input", LaunchConfiguration("input/pointcloud")),
    #             ("output", LaunchConfiguration("output/pointcloud")),
    #         ],
    #         parameters=[
    #             {
    #                 "global_slope_max_angle_deg": 1.5, #10
    #                 "local_slope_max_angle_deg": 1.0, #10
    #                 "split_points_distance_tolerance": 0.05, #0.2
    #                 "split_height_distance": 0.05, #0.2
    #                 "non_ground_height_threshold": 0.15, #0.2
    #                 "gnd_grid_buffer_size": 20, #4
    #                 "grid_size_m": 0.1, #0.5
    #                 "detection_range_z_max": 2.5, #2.5
    #             },
    #             vehicle_info_param,
    #         ],
    #     ),
    # ]

    # nodes = [
    #     ComposableNode(
    #         package="ground_segmentation",
    #         plugin="ground_segmentation::RayGroundFilterComponent",
    #         name="ray_ground_filter",
    #         remappings=[
    #             ("input", LaunchConfiguration("input/pointcloud")),
    #             ("output", LaunchConfiguration("output/pointcloud")),
    #         ],
    #         parameters=[
    #             {
    #                 "general_max_slope": 0.1,
    #                 "local_max_slope": 1.0, 
    #                 "min_height_threshold": 0.2,
    #             },
    #             vehicle_info_param,
    #         ],
    #     ),
    # ] 

        nodes = [
        ComposableNode(
            package="ground_segmentation",
            plugin="ground_segmentation::RingGroundFilterComponent",
            name="ring_ground_filter",
            remappings=[
                ("input", LaunchConfiguration("input/pointcloud")),
                ("output", LaunchConfiguration("output/pointcloud")),
            ],
            parameters=[
                {
                    "variance_judge_vegetation": 0.01, # 0.05
                    "grid_mode_switch_radius": 20.0, #20.0
                    "radial_divider_angle_deg": 2.0, #4.0
                },
                vehicle_info_param,
            ],
        ),
    ] 

    loader = LoadComposableNodes(
        condition=LaunchConfigurationNotEquals("container", ""),
        composable_node_descriptions=nodes,
        target_container=LaunchConfiguration("container"),
    )

    container = ComposableNodeContainer(
        name="scan_ground_filter_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=nodes,
        output="screen",
        condition=LaunchConfigurationEquals("container", ""),
    )

    group = GroupAction(
        [
            container,
            loader,
        ]
    )

    return [group]


def generate_launch_description():
    def add_launch_arg(name: str, default_value=None):
        return DeclareLaunchArgument(name, default_value=default_value)

    default_vehicle_info_param = os.path.join(
        get_package_share_directory("vehicle_info_util"), "config/vehicle_info.param.yaml"
    )

    vehicle_info_param = DeclareLaunchArgument(
        "vehicle_info_param_file",
        default_value=default_vehicle_info_param,
        description="Path to config file for vehicle information",
    )

    return launch.LaunchDescription(
        [
            vehicle_info_param,
            add_launch_arg("container", ""),
            # add_launch_arg("input/pointcloud", "pointcloud"),
            # add_launch_arg("input/pointcloud", "measurement_range/pointcloud"),
            # add_launch_arg("input/pointcloud", "/sensing/lidar/top/outlier_filtered/pointcloud"),
            add_launch_arg("input/pointcloud", "/sensing/lidar/top/pointcloud_raw_ex"),
            add_launch_arg("output/pointcloud", "/ground_filter/pointcloud"),
            # add_launch_arg("output_2/pointcloud", "ground/pointcloud"),
        ]
        + [OpaqueFunction(function=launch_setup)]
    )
