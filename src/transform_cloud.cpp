#include <iostream>
#include <vector> 
#include <cmath> 

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

int main() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud {new pcl::PointCloud<pcl::PointXYZ>}; 

    // Reading a point cloud from .pcd file
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("/home/hamdaan/Dev/probabilistic_ICP/data/room.pcd", *cloud)  == -1){
        PCL_ERROR ("Could not read file room.pcd \n");
        return -1; 
    }

    float theta = 0.0; // M_PI_4/8; 
    std::cout << "Theta: " << theta << std::endl; 

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity(); 
    T.block(0,0,3,3) = Eigen::AngleAxis(theta, Eigen::Vector3f::UnitZ()).toRotationMatrix(); 
    T.block(0,3,3,1) = Eigen::Matrix<float,3,1>(0, 0, 0); 

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ>()); 
    pcl::transformPointCloud(*cloud, *transformed_cloud, T);

    std::cout << "T:\n" << T << std::endl; 

    pcl::io::savePCDFileASCII ("/home/hamdaan/Dev/probabilistic_ICP/data/room_scan_2.pcd", *transformed_cloud);
}