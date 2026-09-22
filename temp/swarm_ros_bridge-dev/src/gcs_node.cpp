//
// Created by bourne on 2024/6/18.
//

#include "drone.h"

Drone drone0, drone1, drone2;



void visCallback(const ros::TimerEvent&event) {
  std::cout << "====================================" << std::endl;
  drone0.coutState();
  drone1.coutState();
  drone2.coutState();
}







int main(int argc, char** argv) {
  ros::init(argc, argv, "gcs_node");
  ros::NodeHandle nh("~");

  drone0.init(nh, 0);
  drone1.init(nh, 1);
  drone2.init(nh, 2);

  ros::Timer visTimer = nh.createTimer(ros::Duration(1.0), visCallback);

  ros::spin();
  return 0;
}



