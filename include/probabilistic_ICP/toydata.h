#include <iostream>
#include <vector> 

#include <cstdlib>

#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

#include <Eigen/Dense>


#ifndef _TOY_DATA
#define _TOY_DATA

std::vector<std::vector<unsigned>> getCorrespondences(int size);
std::vector<Eigen::Matrix<double,-1,1>> toMeasurements(pcl::PointCloud<pcl::PointXY>::Ptr cld);
Eigen::MatrixXd generateCovarianceMatrix(Eigen::VectorXd std); 

#endif