#include <probabilistic_ICP/gauss_newton.h>
#include <cmath>

#include <numeric> 

optim::GaussNewton::GaussNewton(
    std::vector<Eigen::VectorXd> z, // Measurement Vector
    std::vector<Eigen::VectorXd> m, // Previous scan points
    // Ptr to function
    // Arg list: 2D point in previous scan, state 
    Eigen::VectorXd(*obs_model)(Eigen::Vector2d&, Eigen::Vector3d&),
    // Arg list: 2D point in previous scan, current state, previous state
    Eigen::Matrix<double,-1,-1>(*get_obs_jacobi)(Eigen::Vector2d&, Eigen::Vector3d&, Eigen::Vector3d&),
    Eigen::MatrixXd Q, // observation model uncertainty
    Eigen::VectorXd x_prev
) : observation_model(obs_model), get_obs_jacobian(get_obs_jacobi), z_arr(z), m_arr(m), Q(Q), x_t0(x_prev) {

}

optim::GaussNewton::objective_func(
    Eigen::Vector3d x,               // state
    Eigen::Vector3d mew,             // Predicted (prior) state via motion model
    Eigen::Matrix<double,3,3> S_t0,  // Corrected previous state covariance 
    Eigen::Matrix<double,3,3> S_t1_, // Predicted state covariance after motion model is applied
) {

    // H = get_obs_jacobian()
    // S_z = S_t0

    std::vector<unsigned> meas_idx(z.size()); 
    std::iota(meas_idx.begin(), meas_idx.end(), 0); 

    auto measurement_part = [&z_arr, &m_arr, &S_t0, Q, x, x_t0](unsigned int i) {
        Eigen::VectorXd z_i = z_arr.at(i); 
        Eigen::VectorXd m_i = m_arr.at(i); 

        H = get_obs_jacobian(m_i, x, x_t0); 
        S_z = H*S_t0*H.transpose() + Q;
        auto z_est = observation_model(m_i, x, x_t0);

        double res = (z_i - z_est).transpose() * S_z.inverse() * (z_i - z_est); 

        return res; 
    }

    double cost1 = std::accumulate(meas_idx.begin(), meas_idx.end(), 0.0, measurement_part); 

    double cost2 = (mew - x).transpose() * S_t1_.inverse() * (mew - x); 

    return cost1 + cost2; 

}