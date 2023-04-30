#include <iostream>
#include <vector>

#include <Eigen/Dense>

namespace optim {

    class GaussNewton {
        public: 
            GaussNewtom(
                std::vector<Eigen::Vector<double,2> z, // Measurement Vector
                std::vector<Eigen::Vector<double,2> m, // Previous scan points
                // Ptr to function
                // Arg list: 2D point in previous scan, state 
                Eigen::Vector<double,2>(*obs_model)(Eigen::Vector2d&, Eigen::Vector3d&),
                Eigen::Matrix<double,2,2> Q, // observation model uncertainty
            );

            double objective_func(
                Eigen::Vector3d x,               // state
                Eigen::Vector3d mew,             // Predicted state after applying 
                Eigen::Matrix<double,3,3> S_t0,  // Corrected previous state covariance 
                Eigen::Matrix<double,3,3> S_t1_, // Predicted state covariance after motion model is applied
            ); 

            Eigen::Matrix<double,2,3> get_obs_jacobian(Eigen::Vector2d& m, Eigen::Vector3d& x, Eigen::Vector3d& mew); 
    }; 

}