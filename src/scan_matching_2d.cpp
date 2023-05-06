#include <iostream>
#include <vector> 
#include <cmath> 
#include <string>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

Eigen::Matrix<double,2,3> obs_jacobian(Eigen::Vector2d& m, Eigen::Vector3d& x, Eigen::Vector3d& prev_x); 

struct Correspondences {
    std::vector<unsigned> c_idx_1; 
    std::vector<unsigned> c_idx_2; 
    Correspondences(std::vector<unsigned>& vec1, std::vector<unsigned>& vec2) : c_idx_1(vec1), c_idx_2(vec2) {}; 
};

Correspondences getCorrespondences(int size){
    // Everypoint in scan_1 is corresponding to every point in scan_2 in ascending order.
    // Since it is a toy dataset. 
    std::vector<unsigned> scan_1_cor_idx (size); 
    std::vector<unsigned> scan_2_cor_idx (size); 
    std::iota(scan_1_cor_idx.begin(), scan_1_cor_idx.end(), 0);
    std::iota(scan_1_cor_idx.begin(), scan_1_cor_idx.end(), 0);

    return Correspondences(scan_1_cor_idx, scan_2_cor_idx); 
}

std::vector<Eigen::Vector<double,2>> toMeasurements(pcl::PointCloud<pcl::PointXY>::Ptr cld){
    /* 
    This function is used to convert points (x,y) to measurements (range, bearing). 
    Ideally, if you're using a 2D lidar, your measurements would be of the latter form. 
    Just for the sake of this experiment, we will converting a bunch of points (x,y) to 
    the form (range, bearing) to simulate measurement data. 
    */
   std::vector<Eigen::Vector<double,2>> out; 
    for (const auto& p : *cld) { 
        double range = std::sqrt(pow(p.x,2) + pow(p.y,2));
        double theta = std::atan2(p.y, p.x); 
        Eigen::Vector<double,2> v;
        v << range, theta; 
        out.push_back(v); 
    }

    return out; 
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
    Eigen::Matrix3d S_t0; 
    S_t0 << 
    100, 80, 50,
    80, 100, 50,
    50, 50, 25; 

    auto S_t0_inv = S_t0.inverse();

    // Prior uncertainty of current (propagated) state
    Eigen::Matrix3d S_t1_; 
    Eigen::Matrix3d Q; // Motion model uncertainty
    Q <<
    59, 47, 23,
    47, 62, 31,
    23, 31, 14; 

    // Point correspondences. Assuming known correspondences. 
    Correspondences c = getCorrespondences(scan_1->size()); 

    // Converting points in scan_2 to measurement form: (range, bearing)
    const auto z_arr = toMeasurements(scan_2); 

    for (int i = 0; i < z_arr.size(); i++){
        std::cout << z_arr[i][0] << " " << z_arr[i][1]*180/M_PI << std::endl; 
    }

    
}

Eigen::Matrix<double,2,3> obs_jacobian(Eigen::Vector2d p_w, Eigen::Vector3d x){

    Eigen::Matrix2d R_w_b; // rotation which transforms a point from body to world
    R_w_b << 
    cos(x(2)), -sin(x(2)),
    sin(x(2)), cos(x(2));

    Eigen::Vector2d t_w_b; // translation
    t_w_b << x(0), x(1); 

    Eigen::Vector2d p_b = R_w_b.transpose()*(p_w - t_w_b); 

    // Finding partial differentials for computing Jacobian

    double theta = x(2); 

    double dow_p_bx_by_dow_x = -cos(theta); 
    double dow_p_bx_by_dow_y = -sin(theta); 
    double dow_p_bx_by_dow_theta = (p_w[0]-x(0))*sin(theta) - (p_w(1)-x(1))*cos(theta); 

    double dow_p_by_by_dow_x = sin(theta); 
    double dow_p_by_by_dow_y = -cos(theta);
    double dow_p_by_by_dow_theta = (p_w[0]-x(0))*cos(theta) + (p_w(1)-x(1))*sin(theta); 

    double dow_h1_by_dow_p_bx = p_b(0) * pow(pow(p_b(0),2) + pow(p_b(1),2), -0.5); 
    double dow_h1_by_dow_p_by = p_b(1) * pow(pow(p_b(0),2) + pow(p_b(1),2), -0.5); 

    double dow_h2_by_dow_p_bx = -p_b(1) / (pow(p_b(0),2) + pow(p_b(1),2)); 
    double dow_h2_by_dow_p_by = p_b(0) / (pow(p_b(0),2) + pow(p_b(1),2)); 

    double dow_h1_by_dow_x = dow_h1_by_dow_p_bx*dow_p_bx_by_dow_x + dow_h1_by_dow_p_by*dow_p_by_by_dow_x; // J(0,0)
    double dow_h1_by_dow_y = dow_h1_by_dow_p_bx*dow_p_bx_by_dow_y + dow_h1_by_dow_p_by*dow_p_by_by_dow_y; // J(0,1)
    double dow_h1_by_dow_theta = dow_h1_by_dow_p_bx*dow_p_bx_by_dow_theta + dow_h1_by_dow_p_by*dow_p_by_by_dow_theta; // J(0,2)

    double dow_h2_by_dow_x = dow_h2_by_dow_p_bx*dow_p_bx_by_dow_x + dow_h2_by_dow_p_by*dow_p_by_by_dow_x; // J(1,0)
    double dow_h2_by_dow_y = dow_h2_by_dow_p_bx*dow_p_bx_by_dow_y + dow_h2_by_dow_p_by*dow_p_by_by_dow_y; // J(1,1)
    double dow_h2_by_dow_theta = dow_h2_by_dow_p_bx*dow_p_bx_by_dow_theta + dow_h2_by_dow_p_by*dow_p_by_by_dow_theta; // J(2,2)

    Eigen::Matrix<double,2,3> H; // Jacobian
    H  << 
    dow_h1_by_dow_x, dow_h1_by_dow_y, dow_h1_by_dow_theta,
    dow_h2_by_dow_x, dow_h2_by_dow_y, dow_h2_by_dow_theta; 

    return H; 
}

Eigen::Vector2d obs_model(Eigen::Vector2d p_w, Eigen::Vector3d x) {

    Eigen::Matrix2d R_w_b; // rotation which transforms a point from body to world
    R_w_b << 
    cos(x(2)), -sin(x(2)),
    sin(x(2)), cos(x(2));

    Eigen::Vector2d t_w_b; // translation
    t_w_b << x(0), x(1); 

    Eigen::Vector2d p_b = R_w_b.transpose()*(p_w - t_w_b); 

    double range = std::sqrt(pow(p_b(0),2) + pow(p_b(1),2)); 
    double theta = atan2(p_b(1), p_b(0)); 

    Eigen::VectorXd v(2); 
    v << range, theta; 

    return v; 
}