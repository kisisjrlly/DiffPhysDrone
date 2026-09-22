#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>

// generate a big slope
pcl::PointCloud<pcl::PointXYZ>::Ptr generatePlane(double width, double height, double resolution)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for(double x = 0; x < width; x += resolution)
    {
        for(double y = 0; y < height; y += resolution)
        {
            pcl::PointXYZ point;
            point.x = x;
            point.y = y;
            point.z = y / tan(45 * M_PI / 180.0); // 45 degrees with x-y plane
            cloud->points.push_back(point);
        }
    }

    // add a transform the move the cube to the (0,0,0) center
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << -width / 2.0, -height / 2.0, -height /2.0/ tan(45 * M_PI / 180.0);
    pcl::transformPointCloud(*cloud, *cloud, transform);

    cloud->width = (int) cloud->points.size();
    cloud->height = 1;

    return cloud;
}

// generate a big empty cube
pcl::PointCloud<pcl::PointXYZ>::Ptr generateCube(double width, double length, double height, double res)
{
    // according to width, length and height, generate a cube surface
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for(double x = 0; x < width; x += res)
    {
        for(double y = 0; y < length; y += res)
        {
            pcl::PointXYZ point1, point2;
            point1.x = x;
            point1.y = y;
            point1.z = 0.0;
            point2.x = x;
            point2.y = y;
            point2.z = height;
            cloud->points.push_back(point1);
            cloud->points.push_back(point2);
        }
    }

    for(double x = 0; x < width; x += res)
    {
        for(double z = 0; z < height; z += res)
        {
            pcl::PointXYZ point1, point2;
            point1.x = x;
            point1.y = 0.0;
            point1.z = z;
            point2.x = x;
            point2.y = length;
            point2.z = z;
            cloud->points.push_back(point1);
            cloud->points.push_back(point2);
        }
    }

    for(double y = 0; y < length; y += res)
    {
        for(double z = 0; z < height; z += res)
        {
            pcl::PointXYZ point1, point2;
            point1.x = 0.0;
            point1.y = y;
            point1.z = z;
            point2.x = width;
            point2.y = y;
            point2.z = z;
            cloud->points.push_back(point1);
            cloud->points.push_back(point2);
        }
    }

    // add a transform the move the cube to the (0,0,0) center
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << -width / 2.0, -length / 2.0, -height / 2.0;
    pcl::transformPointCloud(*cloud, *cloud, transform);

    cloud->width = (int) cloud->points.size();
    cloud->height = 1;

    return cloud;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "plane_pcd_generator");
    ros::NodeHandle nh;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = generatePlane(300.0, 300.0, 0.07);

    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = generateCube(200.0, 50.0, 50.0, 0.07);

    if(pcl::io::savePCDFileASCII("/home/jackykong/largescale_multiexplore_ws/src/largescale_multi_exploration/generates_pcds/pcd/big_box.pcd", *cloud) == -1)
    {
        ROS_ERROR("Failed to write the PCD file");
        return -1;
    }

    ROS_INFO("Saved the PCD file as test_pcd.pcd");

    return 0;
}
