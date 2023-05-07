#include <probabilistic_ICP/toydata.h> 

std::vector<std::vector<unsigned>> getCorrespondences(int size){
    // Everypoint in scan_1 is corresponding to every point in scan_2 in ascending order.
    // Since it is a toy dataset. 

    std::vector<std::vector<unsigned>> c; 
    for (unsigned i = 0; i < size; i++){
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

Eigen::MatrixXd generateCovarianceMatrix(Eigen::VectorXd mean, Eigen::VectorXd std){
    int samples = 10; 

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(mean.rows(), mean.rows()); 

    for (int n = 0; n < samples; n++){

        Eigen::VectorXd x_i(3); 
        for (int i = 0; i < mean.rows(); i++){
            // r is a random floating point number between 0.0 and 1.0, inclusive
            double r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
            double x = (mean[i]-std[i]) + (r * 2*std[i]); 
            x_i[i] = x; 
        } 

        // error matrix
        Eigen::MatrixXd E = (x_i - mean) * (x_i - mean).transpose(); 

        C = C+E; 
    }

    C = 1/((double)samples-1) * C; 

    return C; 
}