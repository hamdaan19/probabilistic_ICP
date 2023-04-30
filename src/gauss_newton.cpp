#include <probabilistic_ICP/guass_newton.h>
#include <cmath>

Eigen::Matrix<double,2,3> optim::GaussNewton::get_obs_jacobian(Eigen::Vector2d& m, Eigen::Vector3d& x, Eigen::Vector3d& mew){
    dx = x - mew; 

    Eigen::Matrix2d dR << // relative rotation
    cos(dx(2)), -sin(dx(2)),
    sin(dx(2)), cos(dx(2));

    Eigen::Vector2d dt << dx(0), dx(1); // relative translation

    Eigen::Matrix3d dT_rev = Eigen::Matrix3d::Identity(); // relative reverse transformation
    dT_rev.block(0,0,2,2) = dR.tranpose(); 
    dT_rev.block(0,2,2,1) = -dt; 

    Eigen::Vector3d m_homo << m(0), m(1), 1.0; // previous scan corresponding point in homogeneous coordinates

    auto m_cap_homo = dT_rev*m_homo; 

    double dow_h1_by_dow_x = -1 * m_cap_homo(0) * pow( pow(m_cap_homo(0),2) + pow(m_cap_homo(1),2), -0.5); 
    double dow_h1_by_dow_y = -1 * m_cap_homo(1) * pow( pow(m_cap_homo(0),2) + pow(m_cap_homo(1),2), -0.5); 
    double dow_h1_by_dow_theta = pow( pow(m_cap_homo(0),2) + pow(m_cap_homo(1),2), -0.5) * (
        m_cap_homo(0) * (m_homo(0)*sin(dx(2)) - m_homo(1)*cos(dx(2))) +
        m_cap_homo(1) * (m_homo(0)*cos(dx(2)) + m_homo(1)*sin(dx(2))) ) ;

     




}