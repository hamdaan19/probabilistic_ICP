#include <iostream>
#include <vector>

#include <Eigen/Dense>


namespace optim {

    class GaussNewton {
        public: 
            GaussNewton(
                std::vector<Eigen::VectorXd> z, // Measurement Vector
                std::vector<Eigen::VectorXd> p_w, // Previous scan points
                // Elements in c are indices of first and second scans 'respectively'. 
                std::vector<std::vector<unsigned>> c, // Correspondences
                // Ptr to function
                // Arg list: 2D point in previous scan, state 
                Eigen::VectorXd(*obs_model)(Eigen::VectorXd&, Eigen::VectorXd&),
                // Arg list: 2D point in previous scan, current state, previous state
                Eigen::Matrix<double,-1,-1>(*get_obs_jacobi)(Eigen::VectorXd, Eigen::VectorXd),
                Eigen::MatrixXd Q, // observation model uncertainty
                Eigen::MatrixXd S_prev, // Previoud time-step posterior state covariance 
                Eigen::VectorXd x_pred, // Next time-step prior (predicted) state
                Eigen::MatrixXd S_pred // Next time-step prior (predicted) state covariance 
            );

            Eigen::VectorXd(*observation_model)(Eigen::VectorXd&, Eigen::VectorXd&); 
            Eigen::Matrix<double,-1,-1>(*get_obs_jacobian)(Eigen::VectorXd, Eigen::VectorXd); 
            std::vector<std::vector<unsigned>> c_arr; 
            std::vector<Eigen::VectorXd> z_arr; 
            std::vector<Eigen::VectorXd> p_w_arr;
            Eigen::MatrixXd Q; 

            Eigen::MatrixXd S_t0;
            Eigen::VectorXd x_t1_; 
            Eigen::MatrixXd S_t1_; 

            double objective_func(Eigen::VectorXd x); 

            Eigen::VectorXd optimize(Eigen::Vector3d x_init, double step_length); 
    }; 

}