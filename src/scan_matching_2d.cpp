#include <iostream>
#include <vector> 
#include <cmath> 
#include <string>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

#include <probabilistic_ICP/gauss_newton.h>

using namespace optim; 

Eigen::Matrix<double,-1,-1> obs_jacobian(Eigen::VectorXd p_w, Eigen::VectorXd x); 
Eigen::VectorXd obs_model(Eigen::VectorXd& p_w, Eigen::VectorXd& x);

std::vector<std::vector<unsigned>> getCorrespondences(int size){
    // Everypoint in scan_1 is corresponding to every point in scan_2 in ascending order.
    // Since it is a toy dataset. 

    std::vector<std::vector<unsigned>> c; 
    for (unsigned i = 0; i <= size; i++){
        std::vector<unsigned> v{i, i}; 
        c.push_back(v); 
    }

    return c; 
}

std::vector<Eigen::Vector<double,-1>> toMeasurements(pcl::PointCloud<pcl::PointXY>::Ptr cld){
    /* 
    This function is used to convert points (x,y) to measurements (range, bearing). 
    Ideally, if you're using a 2D lidar, your measurements would be of the latter form. 
    Just for the sake of this experiment, we will converting a bunch of points (x,y) to 
    the form (range, bearing) to simulate measurement data. 
    */
   std::vector<Eigen::Vector<double,-1>> out; 
    for (const auto& p : *cld) { 
        double range = std::sqrt(pow(p.x,2) + pow(p.y,2));
        double theta = std::atan2(p.y, p.x); 
        Eigen::Vector<double,-1> v(2);
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
    10, 5, 2*0.087,
    5, 10, 3*0.087,
    2*0.087, 3*0.087, 0.087*0.087; 

    auto S_t0_inv = S_t0.inverse();

    // Prior uncertainty of current (propagated) state
    Eigen::Matrix3d S_t1_; 
    S_t1_ <<
    200, 180, 13.22*0.523,
    180, 200, 13*0.523,
    13.22*0.523, 13*0.523, 0.523*0.523;

    Eigen::Matrix3d Q; // Motion model uncertainty
    Q <<
    59, 47, 23,
    47, 62, 31,
    23, 31, 14; 

    // Predicted state
    Eigen::Vector3d x_pred;
    x_pred << 1.5, 3.2, 0.22; // true values: 1.1, 3.4, 0.196 

    // Point correspondences. Assuming known correspondences. 
    std::vector<std::vector<unsigned>> c = getCorrespondences(scan_1->size()); 

    // Converting points in scan_2 to measurement form: (range, bearing)
    std::vector<Eigen::VectorXd> z_arr = toMeasurements(scan_2); 

    // Manually casting pcl::PointCloud<pcl::PointXY>::Ptr to std::vector<Eigen::Vector2d> 
    std::vector<Eigen::VectorXd> p_w_arr; 
    for (const auto& point : *scan_1){
        Eigen::VectorXd p_vec(2); 
        p_vec << point.x, point.y; 
        p_w_arr.push_back(p_vec); 
    }

    // Creating a GaussNewton object 
    GaussNewton gn(z_arr, p_w_arr, c, obs_model, obs_jacobian, Q, S_t0, x_pred, S_t1_);

    double val = gn.objective_func(x_pred);  

    std::cout << "Val: " << val << std::endl; 

    
}

Eigen::Matrix<double,-1,-1> obs_jacobian(Eigen::VectorXd p_w, Eigen::VectorXd x){

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

    Eigen::Matrix<double,-1,-1> H(2,3); // Jacobian
    H  << 
    dow_h1_by_dow_x, dow_h1_by_dow_y, dow_h1_by_dow_theta,
    dow_h2_by_dow_x, dow_h2_by_dow_y, dow_h2_by_dow_theta; 

    return H; 
}

Eigen::VectorXd obs_model(Eigen::VectorXd& p_w, Eigen::VectorXd& x) {

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