# real-flight-utils

## 2023.12.26

real-flight-utils-lidar
具体内容：

- **FAST_LIO**：雷达与imu融合的slam，具体为fastlio2，输出的频率较低，为10hz
- **ekf_pose**：扩展卡尔曼滤波，用来提高激光雷达的定位频率，用以进行控制，提高到了100hz
- **px4ctrl**：控制节点，用offboard实现了一个位置控制

剩下两个包是驱动