#include <probabilistic_ICP/gauss_newton.h>
#include <cmath>

#include <numeric> 

optim::GaussNewton::GaussNewton(
    std::vector<Eigen::VectorXd> z, // Measurement Vector
    std::vector<Eigen::VectorXd> p_w, // Previous scan points in world frame
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
) : observation_model(obs_model), get_obs_jacobian(get_obs_jacobi), c_arr(c),
    z_arr(z), p_w_arr(p_w), Q(Q), S_t0(S_prev), x_t1_(x_pred), S_t1_(S_pred) {

}

double optim::GaussNewton::objective_func(Eigen::VectorXd x) {

    auto measurement_part = [this, &x](double prev, std::vector<unsigned> c_i) { // &this->z_arr, &m_arr, S_t0, Q, x, x_t
        unsigned int scan1idx = c_i[0]; // index of the corresponding point in scan 1
        unsigned int scan2idx = c_i[1]; // index of the corresponding measurement in scan 2

        Eigen::VectorXd z_i = z_arr.at(scan2idx); 
        Eigen::VectorXd p_w_i = p_w_arr.at(scan1idx); 

        Eigen::Matrix<double,-1,-1> H = get_obs_jacobian(p_w_i, x); 
        Eigen::MatrixXd S_z = H*S_t0*H.transpose() + Q;
        Eigen::VectorXd z_pred = observation_model(p_w_i, x);

        double res = (z_i - z_pred).transpose() * S_z.inverse() * (z_i - z_pred); 

        return res+prev; 
    };

    double cost1 = std::accumulate(c_arr.begin(), c_arr.end(), 0.0, measurement_part); 

    double cost2 = (x_t1_ - x).transpose() * S_t1_.inverse() * (x_t1_ - x); 

    std::cout << "cost1: " << cost1 << " cost2: " << cost2 << std::endl; 

    return cost1 + cost2; 

}

Eigen::VectorXd optim::GaussNewton::optimize(Eigen::Vector3d x_init, double step_length) {

    // Non-linear function which is linearized
    auto func_F = [this](Eigen::VectorXd z_i, Eigen::VectorXd p_w_i, Eigen::VectorXd x){
        Eigen::VectorXd z_pred = observation_model(p_w_i, x);
        Eigen::VectorXd f = z_i - z_pred; 
        return f;
    };

    // Function to compute measurement-associated covariance
    auto func_S_z_inv = [this](Eigen::VectorXd p_w_i, Eigen::Vector3d x){
        Eigen::Matrix<double,-1,-1> H = get_obs_jacobian(p_w_i, x); 
        Eigen::MatrixXd S_z = H*S_t0*H.transpose() + Q;
        Eigen::MatrixXd S_z_inv = S_z.inverse(); 
        return S_z_inv; 
    };

    // Giving equal weightage to all correspondences
    double corres_weight = 1/z_arr.size(); 

    Eigen::VectorXd x_k = x_init; 
    int iter = 0; 
    while (iter < 100) { // some condition
        
        Eigen::VectorXd del_x_resultant = Eigen::VectorXd::Zero( x_init.rows() ); 
        for (int i = 0; i < z_arr.size(); i++){

            auto c_i = c_arr.at(i); 
            unsigned int scan1idx = c_i[0];
            unsigned int scan2idx = c_i[1];

            Eigen::VectorXd z_i = z_arr.at(scan2idx); 
            Eigen::VectorXd p_w_i = p_w_arr.at(scan1idx); 
    
            Eigen::Matrix<double,-1,-1> H_i = get_obs_jacobian(p_w_i, x_k); 

            Eigen::MatrixXd S_z_i_inv = func_S_z_inv(p_w_i, x_k);
            Eigen::VectorXd F_i = func_F(z_i, p_w_i, x_k); 

            Eigen::MatrixXd I = Eigen::MatrixXd::Identity(x_init.rows(), x_init.rows()); 
            Eigen::VectorXd del_x_i = ( ( H_i.transpose()*S_z_i_inv*H_i + 0.5*I).inverse() * H_i.transpose()*S_z_i_inv*F_i );
            // std::cout << ( H_i.transpose()*S_z_i_inv*H_i ).inverse()  << std::endl; 
            // std::cout << "-----------------\n";
            del_x_resultant += del_x_i;

        }

        x_k = x_k + (step_length * del_x_resultant); // Updating x per iteration
        std::cout << x_k.transpose() << std::endl; 
        iter++;
    }

    return x_k; 

}