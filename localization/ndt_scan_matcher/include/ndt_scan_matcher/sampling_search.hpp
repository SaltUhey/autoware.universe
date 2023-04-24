#ifndef NDT_SCAN_MATCHER__SAMPLING_SEARCH_HPP_
#define NDT_SCAN_MATCHER__SAMPLING_SEARCH_HPP_

#include "tf2_eigen/tf2_eigen.hpp"

#include <multigrid_pclomp/multigrid_ndt_omp.h>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <array>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "ndt_scan_matcher/util_func.hpp"

// this function checks transformation probability(TP) on trajectory.
// normally, converged position indicates maximum TP.
// however, in tunnel, another place indicates larger TP sometime due to small convergence area.
// this function find maxium TP place on trajectory and update initial position to recover to
// correct position.

struct Sampling_search
{
  using PointSource = pcl::PointXYZ;
  using PointTarget = pcl::PointXYZ;
  using NormalDistributionsTransform =
    pclomp::MultiGridNormalDistributionsTransform<PointSource, PointTarget>;

  Sampling_search() { mc_slip_pose_ = Eigen::Matrix4f::Identity(4, 4); }
  
  //${try_count} times, randomly scatter the points, and if the score is higher than ${er than ${ndt_result.transform_probability}, adopt
  /*void sampling_search(*/
  /*std::vector<pclomp::NdtResult> sampling_search(*/
  struct Compare_pose sampling_search(
    Eigen::Matrix4f result_pose_matrix, double peak_tp,
    std::shared_ptr<pcl::PointCloud<PointSource>> sensor_points_baselinkTF_ptr,
    const std::shared_ptr<NormalDistributionsTransform> & ndt_ptr)
  {
    // getMaxiteration
    int max_iter = ndt_ptr->getMaximumIterations();
    // setMaxiteration to 1
    ndt_ptr->setMaximumIterations(1);
    double mc_max_tp = peak_tp;//peak_tp:origin tp

    auto output_cloud = std::make_shared<pcl::PointCloud<PointSource>>();
    // for (double dist = -2; dist <= 2; dist+=0.5) {
    double dist;
    //add 20230322-----
    //std::vector<pclomp::NdtResult> vec_ndt_canditates;
    Compare_pose compare_pose;
    compare_pose.origin_pose=result_pose_matrix;
    compare_pose.which=0;
    compare_pose.origin_tp=peak_tp;


    //std::cerr << "Here is the matrix result_pose_matrix:\n" << result_pose_matrix << std::endl;

    for (int try_count = 0; try_count < 1; try_count++) {//
      dist = 6.0 * ((double)rand() / RAND_MAX - 0.5);//max 3?
      Eigen::Matrix4f shift_matrix = Eigen::Matrix4f::Identity(4, 4);
      shift_matrix(0, 3) = dist;
      //std::cout << "Here is the matrix shift_matrix:\n" << shift_matrix << std::endl;
      //mc_pose_matrixとresult_pose_matrixのzの出力値を比較する
      const Eigen::Matrix4f mc_pose_matrix = result_pose_matrix * shift_matrix;
      compare_pose.vec_ndt_canditate.push_back(mc_pose_matrix);//実質、tpが高ければmc_pose_matrixの位置が採用されたことになる(?)
      ndt_ptr->setInputSource(sensor_points_baselinkTF_ptr);
      ndt_ptr->align(*output_cloud, mc_pose_matrix);
      double tp = ndt_ptr->getTransformationProbability();
      compare_pose.sampling_tp=tp;
      //add 20230322-----
      //std::vector<pclomp::NdtResult> vec_ndt_canditates;
      //pclomp::NdtResult ndt_canditate = ndt_ptr->getResult();//getResult()でndtやった後
      //vec_ndt_canditates.push_back(ndt_canditate);
      //compare_pose.vec_ndt_canditates_rst.push_back(ndt_canditate);
      //geometry_msgs::msg::Pose camditate_pose_msg = matrix4f_to_pose(ndt_canditate.pose);//publish this
      
      //--------

      if (mc_max_tp < tp) {  // find max tp
        compare_pose.vec_shift.push_back(dist);
        compare_pose.which = 1;
        mc_max_tp = tp;
        mc_slip_pose_ =/*next stepで使う*/
          result_pose_matrix.inverse() * mc_pose_matrix;  //*mc_center_pose_matrix.inverse();
      }
      else{
        dist=0.0;
        compare_pose.vec_shift.push_back(dist);//tpスコアが高いときのみ渡す
      }
    }
    // return to original
    ndt_ptr->setMaximumIterations(max_iter);
    //return vec_ndt_canditates;//std::vector<pclomp::NdtResult> vec_ndt_canditates/std::vector<geometry_msgs::msg::PoseStamped>
    return compare_pose;
  }

  // replace initial position for align process.
  //  if tp_check update slip_pose, then initial pose is updated.
  //  otherwise, initial pose is not changed.
  Eigen::Matrix4f pose_update(Eigen::Matrix4f origin)
  {
    Eigen::Matrix4f new_pose;
    new_pose = origin * mc_slip_pose_;

    mc_slip_pose_ = Eigen::Matrix4f::Identity(4, 4);//new_poseとして、受け渡した後、初期化?
    return new_pose;
  }

  // position offset for initial position
  Eigen::Matrix4f mc_slip_pose_;
};

#endif  // NDT_SCAN_MATCHER__SAMPLING_SEARCH_HPP_