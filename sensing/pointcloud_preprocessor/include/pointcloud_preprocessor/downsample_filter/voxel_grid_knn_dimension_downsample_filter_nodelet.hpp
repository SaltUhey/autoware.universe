#ifndef POINTCLOUD_PREPROCESSOR__DOWNSAMPLE_FILTER__VOXEL_GRID_KNN_DIMENSION_DOWNSAMPLE_FILTER_NODELET_HPP_
#define POINTCLOUD_PREPROCESSOR__DOWNSAMPLE_FILTER__VOXEL_GRID_KNN_DIMENSION_DOWNSAMPLE_FILTER_NODELET_HPP_

#include "pointcloud_preprocessor/filter.hpp"
#include "pointcloud_preprocessor/transform_info.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl/search/pcl_search.h>

#include <vector>

namespace pointcloud_preprocessor
{
class VoxelGridKnnDimensionDownsampleFilterComponent : public pointcloud_preprocessor::Filter
{
protected:
  void filter(
    const PointCloud2ConstPtr & input, const IndicesPtr & indices, PointCloud2 & output) override;

  // // TODO(atsushi421): Temporary Implementation: Remove this interface when all the filter nodes
  // // conform to new API
  // virtual void faster_filter(
  //   const PointCloud2ConstPtr & input, const IndicesPtr & indices, PointCloud2 & output,
  //   const TransformInfo & transform_info);

private:
  float voxel_size_x_;
  float voxel_size_y_;
  float voxel_size_z_;

  /** \brief Parameter service callback result : needed to be hold */
  OnSetParametersCallbackHandle::SharedPtr set_param_res_;

  /** \brief Parameter service callback */
  rcl_interfaces::msg::SetParametersResult paramCallback(const std::vector<rclcpp::Parameter> & p);

public:
  PCL_MAKE_ALIGNED_OPERATOR_NEW
  explicit VoxelGridKnnDimensionDownsampleFilterComponent(const rclcpp::NodeOptions & options);
};
}  // namespace pointcloud_preprocessor

#endif  // POINTCLOUD_PREPROCESSOR__DOWNSAMPLE_FILTER__VOXEL_GRID_KNN_DIMENSION_DOWNSAMPLE_FILTER_NODELET_HPP_
