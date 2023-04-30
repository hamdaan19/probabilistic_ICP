#include <iostream>
#include <vector> 
#include <cmath> 
#include <string>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

struct Correspondences {
    std::vector<unsigned> c_idx_1; 
    std::vector<unsigned> c_idx_2; 
    Correspondences(std::vector<unsigned> &vec1, std::vector<unsigned> &vec2) : c_idx_1(vec1), c_idx_2(vec2) {}; 
};

Correspondences getCorrespondences(int size){
    // Everypoint in scan_1 is corresponding to every point in scan_2 in ascending order.
    // Since it is a toy dataset. 
    std::vector<unsigned> scan_1_cor_idx (size); 
    std::vector<unsigned> scan_2_cor_idx (size); 
    std::iota(scan_1_cor_idx.begin(), scan_1_cor_idx.end(), 0);
    std::iota(scan_1_cor_idx.begin(), scan_1_cor_idx.end(), 0);

    return Correspondences(scan_1_idx, scan_2_idx); 
}

std::vector<Eigen::Vector<double,2>> toMeasurements(pcl::PointCloud<pcl::PointXY>::Ptr cld){
    /* 
    This function is used to convert points (x,y) to measurements (range, bearing). 
    Ideally, if you're using a 2D lidar, your measurements would be of the latter form. 
    Just for the sake of this experiment, we will converting a bunch of points (x,y) to 
    the form (range, bearing) to simulate measurement data. 
    */
   std::vector<Eigen::Vector<double,2>> out; 
    for (const auto& p : cld) { 
        double range = std::sqrt(pow(p.x,2) + pow(p.y,2));
        double theta = std::atan2(p.y, p.x); 
        Eigen::Vector<double,2> v << range, theta; 
        out.push_back(v); 
    }

    return v; 
}

int main(int argc, char* argv[]){

    pcl::PointCloud<pcl::PointXY>::Ptr scan_1 (new pcl::PointCloud<pcl::PointXY>);
    pcl::PointCloud<pcl::PointXY>::Ptr scan_2 (new pcl::PointCloud<pcl::PointXY>);

    if (pcl::io::loadPCDFile<pcl::PointXY>("/home/hamdaan/Dev/probabilistic_ICP/data/room_scan_1.pcd", *scan_1)  == -1){
        PCL_ERROR ("Could not read file room.pcd \n");
        return -1; 
    }    

    if (pcl::io::loadPCDFile<pcl::PointXY>("/home/hamdaan/Dev/probabilistic_ICP/data/room_scan_2.pcd", *scan_2)  == -1){
        PCL_ERROR ("Could not read file room.pcd \n");
        return -1; 
    }  

    // State vector contains x, y, theta. 
    // Previous time step state uncertainty
    Eigen::Matrix3d S_t0_post; 
    S_t0_post << 
    100, 80, 50,
    80, 100, 50,
    50, 50, 25; 

    auto S_t0_post_inv = S_t0_post.inverse();

    // Prior uncertainty of current (propagated) state
    Eigen::Matrix3d S_t1_prior; 
    Eigen::Matrix3d R_t; // Motion model uncertainty
    R_t <<
    59, 47, 23,
    47, 62, 31,
    23, 31, 14; 

    // Point correspondences. Assuming known correspondences. 
    Correspondences c = getCorrespondences(scan_1->size()); 

    // Converting points in scan_2 to measurement form: (range, bearing)
    const auto z_arr = toMeasurements(scan_2); 

    
}