//
// Created by bourne on 2024/6/18.
//
#include "drone.h"

Drone::Drone() {}

Drone::~Drone() {}

void Drone::init(ros::NodeHandle &nh, int drone_id) {
  drone_id_ = drone_id;
  // todo get offset params
  nh.param("drone"+std::to_string(drone_id)+"/offset_x", offset_[0], 0.0);


  pos_ = Eigen::Vector3d::Zero();

  odom_sub_ = nh.subscribe("/odom" + std::to_string(drone_id), 1, &Drone::odomCallback, this);
  // todo state sub

  path_pub_ = nh.advertise<nav_msgs::Path>("/drone"+std::to_string(drone_id)+"/passed_path", 10);

  path_vis_ = nh.createTimer(ros::Duration(0.1), &Drone::pathVisTimer, this);
}

void Drone::odomCallback(const nav_msgs::OdometryConstPtr &msg) {
  pos_[0] = msg->pose.pose.position.x;
  pos_[1] = msg->pose.pose.position.y;
  pos_[2] = msg->pose.pose.position.z;

  // add new point
  if(path_.empty()) {
    path_.push_back(pos_);
  } else {
    Eigen::Vector3d last_pos_ = path_.back();
    if ((pos_ - last_pos_).norm() > 0.2) {
      path_.push_back(pos_);
    }
  }
}

void Drone::stateCallback(const std_msgs::Int32ConstPtr &msg) {

}

void Drone::pathVisTimer(const ros::TimerEvent &event) {
  if(path_.empty()) return;
  // vis kino path
  nav_msgs::Path ros_path;
  ros_path.header.frame_id = "world";

  for(auto & i : path_){
    geometry_msgs::PoseStamped pt;
    pt.pose.position.x = i(0);
    pt.pose.position.y = i(1);
    pt.pose.position.z = i(2);
    pt.pose.orientation.x = 0;
    pt.pose.orientation.y = 0;
    pt.pose.orientation.z = 0;
    pt.pose.orientation.w = 1;

    ros_path.poses.push_back(pt);
  }
  path_pub_.publish(ros_path);
}

void Drone::coutState() {
  std::cout << "[Drone" << drone_id_ << "] pos " << pos_.transpose() << " state " << state_string_[state_] << std::endl;
}