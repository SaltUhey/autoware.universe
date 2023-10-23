#include "pointcloud_preprocessor/downsample_filter/voxel_grid_knn_dimension_downsample_filter_nodelet.hpp"

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/segment_differences.h>

#include <vector>

#include <pcl/common/pca.h>
#include <pcl/features/feature.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <time.h>

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
  clock_t start = clock();

  std::scoped_lock lock(mutex_);
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_input(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_input_ds_rep(new pcl::PointCloud<pcl::PointXYZ>); //代表点群
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_input_ds_ref(new pcl::PointCloud<pcl::PointXYZ>); //参照点群
  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_input_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_output_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromROSMsg(*input, *pcl_input);
  
  pcl_input_ds_rep->points.reserve(pcl_input->points.size());
  pcl::VoxelGrid<pcl::PointXYZ> filter_rough;
  filter_rough.setInputCloud(pcl_input);
  // filter.setSaveLeafLayout(true);
  filter_rough.setLeafSize(voxel_size_x_, voxel_size_y_, voxel_size_z_);
  filter_rough.filter(*pcl_input_ds_rep);//粗めのダウンサンプリング後の点群：代表点群

  pcl_input_ds_ref->points.reserve(pcl_input->points.size());
  pcl::VoxelGrid<pcl::PointXYZ> filter_detail;
  filter_detail.setInputCloud(pcl_input);
  // filter.setSaveLeafLayout(true);
  const float ref_voxel_size = 0.3;
  filter_detail.setLeafSize(ref_voxel_size, ref_voxel_size, ref_voxel_size);
  filter_detail.filter(*pcl_input_ds_ref);//細かめのダウンサンプリング後の点群：参照点群
  
  //20231002
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud (pcl_input_ds_ref);
  int K = 50; // K nearest neighbor search
  std::vector<int> pointIdxKNNSearch(K); //must be resized to k a priori
  std::vector<float> pointKNNSquaredDistance(K); //must be resized to k a priori
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr use_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_visualized (new pcl::PointCloud<pcl::PointXYZRGB>);
  
  for(size_t i= 0; i<pcl_input_ds_rep->size(); i++){
    pcl::PointXYZ searchPoint = pcl_input_ds_rep->points[i];
    kdtree.nearestKSearch (searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_surr_1search (new pcl::PointCloud<pcl::PointXYZ>);
    //cloud_surr_1search->push_back(searchPoint);
    
    for (size_t j =0; j < pointIdxKNNSearch.size(); j++){
      if(pointKNNSquaredDistance[i]<5.0){
      cloud_surr_1search->push_back(pcl_input_ds_ref->points[pointIdxKNNSearch[j]]);
      }
    }
    // std::cerr << "cloud_surr_1search size: " << cloud_surr_1search->size() << std::endl;
    //std::cerr << "check" << std::endl;
    

    //cloud_surr_1searchの調整----------------------------------------

    //Removing noise
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud_surr_1search);
    sor.setMeanK (K/2);//The number of points to use for mean distance estimation
    sor.setStddevMulThresh (ref_voxel_size); //Standard deviations threshold
    sor.filter (*cloud_surr_1search);

    //Clustering
    // const double cluster_tolerance = ref_voxel_size; 
    // const int min_cluster_size = 5;
    // pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    // tree->setInputCloud(cloud_surr_1search);
    // std::vector<pcl::PointIndices> cluster_indices; //クラスタリング後のインデックスが格納されるベクトル
    // cluster_indices.clear();
    // pcl::EuclideanClusterExtraction<pcl::PointXYZ> ece;
    // ece.setClusterTolerance(cluster_tolerance);
    // ece.setMinClusterSize(min_cluster_size);
    // /*各クラスタのメンバの最大数を設定*/
    // ece.setMaxClusterSize(cloud_surr_1search->points.size());
    // ece.setSearchMethod(tree);
    // ece.setInputCloud(cloud_surr_1search);
    // /*クラスリング実行*/
    // pcl::ExtractIndices<pcl::PointXYZ> ei;
    // ei.setInputCloud(cloud_surr_1search);
    // ei.setNegative(false);
    // ece.extract(cluster_indices);
    // pcl::PointIndices::Ptr tmp_clustered_indices (new pcl::PointIndices);
    // if(cluster_indices.size()>=1){
    //   *tmp_clustered_indices = cluster_indices[0];
    //   ei.setIndices(tmp_clustered_indices);
    //   ei.filter(*cloud_surr_1search);
    // }
    
    //std::cerr << "cloud_surr_1search size" << cloud_surr_1search->size() << std::endl;
    //-------------------------------------------------------------

    const int judge_num = 10;
    if(cloud_surr_1search->size()>=judge_num){
      //judge cloud_surr_1search(point cloud) feature
      pcl::PCA<pcl::PointXYZ> pca;
      // pcl::PointIndices::Ptr pca_indices(new pcl::PointIndices());
      pca.setInputCloud(cloud_surr_1search);
      //pca.setIndices(pointIdxKNNSearch);//よくわからない
      Eigen::Vector3f& eigen_values = pca.getEigenValues();
      
      //HERE Comupute eigenvalue difference features
      double lam1,lam2,lam3;
      lam1 = eigen_values (0), lam2 = eigen_values (1), lam3 = eigen_values (2);
      // std::cerr << "lam1(eigen_values (0))" << lam1 << std::endl;
      // std::cerr << "lam2(eigen_values (1))" << lam2 << std::endl;
      // std::cerr << "lam3(eigen_values (2))" << lam3 << std::endl;
      const double s1=std::abs(lam1-lam2);
      const double s2=std::abs(lam2-lam3);
      const double s3=std::abs(lam3);
      const double evalue_diff_ftrs[4]={0,s1,s2,s3};
      int d=0;
      for(int i=1;i<=3;i++)
      {
        if(evalue_diff_ftrs[i-1]<evalue_diff_ftrs[i]){d=i;}
      }
      //std::cerr << "dimension: " << d << std::endl;

      // Visualize
      pcl::PointXYZRGB color_point;
      color_point.x = searchPoint.x;
      color_point.y = searchPoint.y;
      color_point.z = searchPoint.z;
      color_point.r = 255;
      color_point.g = 255;
      color_point.b = 255;
      cloud_visualized->push_back(color_point);

      if (d==1){
        Eigen::Matrix3f& eigen_vectors = pca.getEigenVectors();//d=1のときのみでよいかも
        //std::cerr << "Eigenvectors:\n" << eigenvectors << std::endl;
        const Eigen::Vector3f eigenvector_1= eigen_vectors.col(0); //first eigen vector
        const Eigen::Vector3f z_axis(0, 0, 1);
        float angle_Z_rad = std::acos(eigenvector_1.dot(z_axis));
        float angle_Z_deg = angle_Z_rad*(180.0/M_PI);
        //std::cerr << "angle_X [rad]" << angle_X_rad << std::endl;

        if (((angle_Z_deg < 10) || ((170<angle_Z_deg)&&(angle_Z_deg<190))) && sqrt(lam2)<=0.35){
          for (size_t i = 0; i<cloud_surr_1search->size();i++){
            pcl::PointXYZRGB point;
            point.x = cloud_surr_1search->points[i].x;
            point.y = cloud_surr_1search->points[i].y;
            point.z = cloud_surr_1search->points[i].z;
            point.r = 255;
            point.g = 241;
            point.b = 0;
            //cloud_visualized->push_back(point);
            use_cloud->push_back(point);
          }
        }
        else{
          for (size_t i = 0; i<cloud_surr_1search->size();i++){
            // pcl::PointXYZRGB point;
            // point.x = cloud_surr_1search->points[i].x;
            // point.y = cloud_surr_1search->points[i].y;
            // point.z = cloud_surr_1search->points[i].z;
            // point.r = 255;
            // point.g = 0;
            // point.b = 0;
            // cloud_visualized->push_back(point);
            
            pcl::PointXYZRGB point;
            point.x = searchPoint.x;
            point.y = searchPoint.y;
            point.z = searchPoint.z;
            point.r = 255;
            point.g = 0;
            point.b = 0;
            use_cloud->push_back(point);
            
          }
        }

      }
      else if (d==2){
        
        for (size_t i = 0; i<cloud_surr_1search->size();i++){
          // pcl::PointXYZRGB point;
          // point.x = cloud_surr_1search->points[i].x;
          // point.y = cloud_surr_1search->points[i].y;
          // point.z = cloud_surr_1search->points[i].z;
          // point.r = 0;
          // point.g = 100;
          // point.b = 255;
          // cloud_visualized->push_back(point);

          pcl::PointXYZRGB point;
          point.x = searchPoint.x;
          point.y = searchPoint.y;
          point.z = searchPoint.z;
          point.r = 0;
          point.g = 100;
          point.b = 255;
          use_cloud->push_back(point);
        }
      }    
      else if (d==3){
        
        // for (size_t i = 0; i<cloud_surr_1search->size();i++){
        //   pcl::PointXYZRGB point;
        //   point.x = cloud_surr_1search->points[i].x;
        //   point.y = cloud_surr_1search->points[i].y;
        //   point.z = cloud_surr_1search->points[i].z;
        //   point.r = 0;
        //   point.g = 255;
        //   point.b = 0;
        //   cloud_visualized->push_back(point);
        // }
      }
    }

  }
  
  //pcl::toROSMsg(*cloud_visualized, output);
  pcl::toROSMsg(*use_cloud, output);
  output.header = input->header;

  clock_t end = clock();
  const double time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
  printf("time %lf[ms]\n", time);

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
