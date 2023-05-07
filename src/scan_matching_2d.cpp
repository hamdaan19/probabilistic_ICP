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
#include <probabilistic_ICP/toydata.h>

using namespace optim; 

Eigen::Matrix<double,-1,-1> obs_jacobian(Eigen::VectorXd p_w, Eigen::VectorXd x); 
Eigen::VectorXd obs_model(Eigen::VectorXd& p_w, Eigen::VectorXd& x);


int main(int argc, char* argv[]){

    srand(0); 

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
    Eigen::Vector3d x_t0_std;
    x_t0_std << 10.0, 10.0, 0.0523;  
    Eigen::Matrix3d S_t0 = generateCovarianceMatrix(x_t0_std);  

    // Prior uncertainty of current (propagated) state
    Eigen::Vector3d x_pred_std; 
    x_pred_std << 40.0, 40.0, 0.2618;
    Eigen::Matrix3d S_pred = generateCovarianceMatrix(x_pred_std); 

    // Measurement model uncertainty
    Eigen::Vector2d z_std;
    z_std << 2, 0.035; 
    Eigen::Matrix2d Q = generateCovarianceMatrix(z_std); 

    // Predicted state
    Eigen::Vector3d x_pred; // True: -1.1, -3.4, -0.0982
    x_pred << -1.0, -3.5, -0.12; 

    // Point correspondences. Assuming known correspondences. 
    std::vector<std::vector<unsigned>> c = getCorrespondences(scan_1->size()); 

    // Converting points in scan_2 to measurement form: (range, bearing)
    std::vector<Eigen::VectorXd> z_arr = toMeasurements(scan_2); 

    /*  Manually casting pcl::PointCloud<pcl::PointXY>::Ptr to std::vector<Eigen::Vector2d> */
    std::vector<Eigen::VectorXd> p_w_arr; 
    for (const auto& point : *scan_1){
        Eigen::VectorXd p_vec(2); 
        p_vec << point.x, point.y; 
        p_w_arr.push_back(p_vec); 
    }
    /*--------------------------------------------------------------------------------------*/

    // Creating a GaussNewton object 
    GaussNewton gn(z_arr, p_w_arr, c, obs_model, obs_jacobian, Q, S_t0, x_pred, S_pred);

    Eigen::Vector3d x; x << -1.0, -3.5, -0.12; // True: -1.1, -3.4, -0.0982
    double val = gn.objective_func(x);  

    std::cout << "Val: " << val << std::endl; 

    auto x_opt = gn.optimize(x_pred, 0.005); 
    std::cout << x_opt << std::endl; 

    // Eigen::VectorXd std(3); 
    // std << 100, 90, 0.436; 

    // Eigen::VectorXd x_pred_zero = Eigen::VectorXd::Zero(3); 

    // auto covmat = generateCovarianceMatrix(std); 

    // std::cout << "cov: \n" << covmat << std::endl; 

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
    double theta = std::atan2(p_b(1), p_b(0)); 

    Eigen::VectorXd v(2); 
    v << range, theta; 

    return v; 
}