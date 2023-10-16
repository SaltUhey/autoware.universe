#include "pointcloud_preprocessor/downsample_filter/voxel_grid_knn_dimension_downsample_filter_nodelet.hpp"

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/segment_differences.h>

#include <vector>

#include <pcl/common/pca.h>
#include <pcl/features/feature.h>

namespace pointcloud_preprocessor
{
VoxelGridKnnDimensionDownsampleFilterComponent::VoxelGridKnnDimensionDownsampleFilterComponent(
  const rclcpp::NodeOptions & options)
: Filter("VoxelGridKnnDimensionDownsampleFfilter", options)
{
  // set initial parameters
  {
    voxel_size_x_ = static_cast<double>(declare_parameter("voxel_size_x", 0.3));
    voxel_size_y_ = static_cast<double>(declare_parameter("voxel_size_y", 0.3));
    voxel_size_z_ = static_cast<double>(declare_parameter("voxel_size_z", 0.1));
  }

  using std::placeholders::_1;
  set_param_res_ = this->add_on_set_parameters_callback(
    std::bind(&VoxelGridKnnDimensionDownsampleFilterComponent::paramCallback, this, _1));
}

void VoxelGridKnnDimensionDownsampleFilterComponent::filter(
  const PointCloud2ConstPtr & input, const IndicesPtr & /*indices*/, PointCloud2 & output)
{
  std::scoped_lock lock(mutex_);
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_input(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_output(new pcl::PointCloud<pcl::PointXYZ>);
  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_input_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_output_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromROSMsg(*input, *pcl_input);
  //pcl_output->points.reserve(pcl_input->points.size());
  pcl_output->points.reserve(pcl_input->points.size());
  pcl::VoxelGrid<pcl::PointXYZ> filter;
  filter.setInputCloud(pcl_input);
  // filter.setSaveLeafLayout(true);
  filter.setLeafSize(voxel_size_x_, voxel_size_y_, voxel_size_z_);
  filter.filter(*pcl_output);//細かめのダウンサンプリング後の点群：pcl_output
  
  //20231002
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud (pcl_input);
  int K = 100; // K nearest neighbor search
  std::vector<int> pointIdxKNNSearch(K); //must be resized to k a priori
  std::vector<float> pointKNNSquaredDistance(K); //must be resized to k a priori
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr use_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);//そのうちRGBに戻す
  
  for(size_t i= 0; i<pcl_output->size(); i++){
    pcl::PointXYZ searchPoint = pcl_output->points[i];
    kdtree.nearestKSearch (searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_surr_1search (new pcl::PointCloud<pcl::PointXYZ>);
    //cloud_surr_1search->push_back(searchPoint);
    
    for (size_t j =0; j < pointIdxKNNSearch.size(); j++){
      cloud_surr_1search->push_back(pcl_input->points[pointIdxKNNSearch[j]]);
    }
    // std::cerr << "cloud_surr_1search size: " << cloud_surr_1search->size() << std::endl;
    //std::cerr << "check" << std::endl;

    //judge cloud_surr_1search(point cloud) feature
    pcl::PCA<pcl::PointXYZ> pca;
    // pcl::PointIndices::Ptr pca_indices(new pcl::PointIndices());
    pca.setInputCloud(cloud_surr_1search);
    //pca.setIndices(pointIdxKNNSearch);//よくわからない
    Eigen::Vector3f& eigen_values = pca.getEigenValues();

    //HERE Comupute eigenvalue difference features
    double lam1,lam2,lam3;
    lam1 = eigen_values (0), lam2 = eigen_values (1), lam3 = eigen_values (2);
    std::cerr << "lam1(eigen_values (0))" << lam1 << std::endl;
    std::cerr << "lam2(eigen_values (1))" << lam2 << std::endl;
    std::cerr << "lam3(eigen_values (2))" << lam3 << std::endl;
    const double s1=lam1-lam2;
    const double s2=lam2-lam3;
    const double s3=lam3;
    const double evalue_diff_ftrs[4]={0,s1,s2,s3};
    int d=0;
    for(int i=1;i<=3;i++)
    {
      if(evalue_diff_ftrs[i-1]<evalue_diff_ftrs[i]){d=i;}
    }
    std::cerr << "dimension: " << d << std::endl;
    pcl::PointXYZRGB color_point;
    color_point.x = searchPoint.x;
    color_point.y = searchPoint.y;
    color_point.z = searchPoint.z;
    color_point.r = 255;
    color_point.g = 255;
    color_point.b = 255;

    if (d==1){
      use_cloud->push_back(color_point);
      for (size_t i = 0; i<cloud_surr_1search->size();i++){
        pcl::PointXYZRGB point;
        point.x = cloud_surr_1search->points[i].x;
        point.y = cloud_surr_1search->points[i].y;
        point.z = cloud_surr_1search->points[i].z;
        point.r = 255;
        point.g = 0;
        point.b = 0;
        use_cloud->push_back(point);
      }
    }
    else if (d==2){
      use_cloud->push_back(color_point);
      for (size_t i = 0; i<cloud_surr_1search->size();i++){
        pcl::PointXYZRGB point;
        point.x = cloud_surr_1search->points[i].x;
        point.y = cloud_surr_1search->points[i].y;
        point.z = cloud_surr_1search->points[i].z;
        point.r = 0;
        point.g = 100;
        point.b = 255;
        use_cloud->push_back(point);
      }
    }    
    else if (d==3){
      use_cloud->push_back(color_point);
      for (size_t i = 0; i<cloud_surr_1search->size();i++){
        pcl::PointXYZRGB point;
        point.x = cloud_surr_1search->points[i].x;
        point.y = cloud_surr_1search->points[i].y;
        point.z = cloud_surr_1search->points[i].z;
        point.r = 0;
        point.g = 255;
        point.b = 0;
        use_cloud->push_back(point);
      }
    }    
  }
  
  pcl::toROSMsg(*use_cloud, output);
  output.header = input->header;
}

rcl_interfaces::msg::SetParametersResult VoxelGridKnnDimensionDownsampleFilterComponent::paramCallback(
  const std::vector<rclcpp::Parameter> & p)
{
  std::scoped_lock lock(mutex_);

  if (get_param(p, "voxel_size_x", voxel_size_x_)) {
    RCLCPP_DEBUG(get_logger(), "Setting new distance threshold to: %f.", voxel_size_x_);
  }
  if (get_param(p, "voxel_size_y", voxel_size_y_)) {
    RCLCPP_DEBUG(get_logger(), "Setting new distance threshold to: %f.", voxel_size_y_);
  }
  if (get_param(p, "voxel_size_z", voxel_size_z_)) {
    RCLCPP_DEBUG(get_logger(), "Setting new distance threshold to: %f.", voxel_size_z_);
  }

  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";

  return result;
}
}  // namespace pointcloud_preprocessor

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(pointcloud_preprocessor::VoxelGridKnnDimensionDownsampleFilterComponent)
