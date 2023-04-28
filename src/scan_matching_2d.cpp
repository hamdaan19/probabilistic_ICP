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
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("/home/hamdaan/Dev/probabilistic_ICP/src/room.pcd", *cloud)  == -1){
        PCL_ERROR ("Could not read file test_pcd.pcd \n");
        return -1; 
    }

    float theta = M_PI_4/2; 

    Eigen::Matrix4f B = Eigen::Matrix4f::Identity(); 
    B.block(0,0,3,3) = Eigen::AngleAxis(theta, Eigen::Vector3f::UnitZ()).toRotationMatrix(); 
    B.block(0,3,3,1) = Eigen::Vector<float,3>(1.1, 3.4, 0); 

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ>()); 
    pcl::transformPointCloud(*cloud, *transformed_cloud, B);


}