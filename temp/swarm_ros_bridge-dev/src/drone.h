//
// Created by bourne on 2024/6/18.
//
#include <ros/ros.h>
#include <ros/ros.h>
#include <Eigen/Eigen>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Int32.h>


class Drone {
public:
  Drone();
  ~Drone();

  void init(ros::NodeHandle& nh, int drone_id);
  void coutState();

private:
  int drone_id_;
  int state_;
  Eigen::Vector3d pos_;
  Eigen::Vector3d offset_;

  std::vector<Eigen::Vector3d> path_;
  std::vector<std::string> state_string_ = {
      "INIT",
      "TAKE_OFF",
      "H0VER",
      "WAIT_TRIGGER",
      "MOVE!!"
  };

  ros::Subscriber odom_sub_, state_sub_;
  ros::Publisher path_pub_;
  ros::Timer path_vis_;

  void odomCallback(const nav_msgs::OdometryConstPtr& msg);
  void stateCallback(const std_msgs::Int32ConstPtr& msg);

  void pathVisTimer(const ros::TimerEvent& event);
};