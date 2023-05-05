#include <probabilistic_ICP/gauss_newton.h>
#include <cmath>

#include <numeric> 

optim::GaussNewton::GaussNewton(
    std::vector<Eigen::VectorXd> z, // Measurement Vector
    std::vector<Eigen::VectorXd> m, // Previous scan points
    // Ptr to function
    // Arg list: 2D point in previous scan, state 
    Eigen::VectorXd(*obs_model)(Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&),
    // Arg list: 2D point in previous scan, current state, previous state
    Eigen::Matrix<double,-1,-1>(*get_obs_jacobi)(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd),
    Eigen::MatrixXd Q, // observation model uncertainty
    Eigen::VectorXd x_prev, // Previous time-step posterior state estimate 
    Eigen::MatrixXd S_prev, // Previoud time-step posterior state covariance 
    Eigen::VectorXd x_pred, // Next time-step prior (predicted) state
    Eigen::MatrixXd S_pred // Next time-step prior (predicted) state covariance 
) : observation_model(obs_model), get_obs_jacobian(get_obs_jacobi),
    z_arr(z), m_arr(m), Q(Q), x_t0(x_prev), S_t0(S_prev), x_t1_(x_pred), S_t1_(S_pred) {

}

double optim::GaussNewton::objective_func(Eigen::VectorXd x) {

    std::vector<unsigned> meas_idx(z_arr.size()); 
    std::iota(meas_idx.begin(), meas_idx.end(), 0); 

    auto measurement_part = [this, &x](double prev, unsigned i) { // &this->z_arr, &m_arr, S_t0, Q, x, x_t
        Eigen::VectorXd z_i = z_arr.at(i); 
        Eigen::VectorXd m_i = m_arr.at(i); 

        Eigen::Matrix<double,-1,-1> H = get_obs_jacobian(m_i, x, x_t0); 
        Eigen::MatrixXd S_z = H*S_t0*H.transpose() + Q;
        Eigen::VectorXd z_pred = observation_model(m_i, x, x_t0);

        double res = (z_i - z_pred).transpose() * S_z.inverse() * (z_i - z_pred); 

        return res + prev; 
    };

    double cost1 = std::accumulate(meas_idx.begin(), meas_idx.end(), 0, measurement_part); 

    double cost2 = (x_t1_ - x).transpose() * S_t1_.inverse() * (x_t1_ - x); 

    return cost1 + cost2; 

}

Eigen::VectorXd optim::GaussNewton::optimize(Eigen::Vector3d x_init, double step_length) {

    // Non-linear function
    auto func_F = [this](Eigen::VectorXd z_i, Eigen::VectorXd m_i, Eigen::VectorXd x){
        Eigen::VectorXd z_pred = observation_model(m_i, x, x_t0);
        return z_i - z_pred; 
    };

    // Function to compute measurement-associated covariance
    auto func_S_z_inv = [this](Eigen::VectorXd m_i, Eigen::Vector3d x){
        Eigen::Matrix<double,-1,-1> H = get_obs_jacobian(m_i, x, x_t0); 
        Eigen::MatrixXd S_z = H*S_t0*H.transpose() + Q;
        return S_z.inverse(); 
    };

    // Giving equal weightage to all correspondences
    double corres_weight = 1/z_arr.size(); 

    auto x_k = x_init; 
    while (true) { // some condition
        
        Eigen::VectorXd del_x_resultant = Eigen::VectorXd::Zero( x_init.rows() ); 
        for (int i = 0; i < z_arr.size(); i++){

            Eigen::VectorXd z_i = z_arr.at(i); 
            Eigen::VectorXd m_i = m_arr.at(i); 
    
            Eigen::Matrix<double,-1,-1> H_i = get_obs_jacobian(m_i, x_k, x_t0); 

            Eigen::MatrixXd S_z_i_inv = func_S_z_inv(m_i, x_k);
            Eigen::VectorXd F_i = func_F(z_i, m_i, x_k); 

            Eigen::VectorXd del_x_i = corres_weight * ( ( H_i.transpose()*S_z_i_inv*H_i ).inverse() * H_i.transpose()*S_z_i_inv*F_i );

            del_x_resultant += del_x_i;

        }

        x_k = x_k + step_length * del_x_resultant; // Updating x per iteration
    }

    return x_k; 

}