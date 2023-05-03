#include <iostream>
#include <vector>

#include <Eigen/Dense>

namespace optim {

    class GaussNewton {
        public: 
            GaussNewton(
                std::vector<Eigen::VectorXd> z, // Measurement Vector
                std::vector<Eigen::VectorXd> m, // Previous scan points
                // Ptr to function
                // Arg list: 2D point in previous scan, state 
                Eigen::VectorXd(*obs_model)(Eigen::Vector2d&, Eigen::Vector3d&),
                // Arg list: 2D point in previous scan, current state, previous state
                Eigen::Matrix<double,-1,-1>(*get_obs_jacobi)(Eigen::Vector2d&, Eigen::Vector3d&, Eigen::Vector3d&),
                Eigen::MatrixXd Q, // observation model uncertainty
                Eigen::VectorXd x_prev
            );

            Eigen::VectorXd(*observation_model)(Eigen::Vector2d&, Eigen::Vector3d&); 
            Eigen::Matrix<double,-1,-1>(*get_obs_jacobian)(Eigen::Vector2d&, Eigen::Vector3d&, Eigen::Vector3d&); 
            std::vector<Eigen::VectorXd> z_arr; 
            std::vector<Eigen::VectorXd> m_arr;
            Eigen::MatrixXd Q; 
            Eigen::VectorXd x_t0; 

            double objective_func(
                Eigen::Vector3d x,               // state
                Eigen::Vector3d mew,             // Predicted (prior) state via motion model
                Eigen::Matrix<double,3,3> S_t0,  // Corrected previous state covariance 
                Eigen::Matrix<double,3,3> S_t1_, // Predicted state covariance after motion model is applied
            ); 

            Eigen::Matrix<double,2,3> get_obs_jacobian(Eigen::Vector2d& m, Eigen::Vector3d& x, Eigen::Vector3d& mew); 
    }; 

}