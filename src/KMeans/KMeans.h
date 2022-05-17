// Author: Nathan Crosby
// Version: 0.0
// Date: 20210301
// ChangeLog:
//     0.0 -> initial creation

#ifndef KMEANS_H
#define KMEANS_H

namespace KMeans {
  Eigen::MatrixXd calcDist(Eigen::MatrixXd &, Eigen::MatrixXd &);
  int diffCat(Eigen::MatrixXd &, Eigen::MatrixXd &);
  void calcCenterMean(Eigen::MatrixXd &, Eigen::MatrixXd &, Eigen::MatrixXd &);
  Eigen::MatrixXd kmeans(Eigen::MatrixXd &, double, Eigen::MatrixXd &);
} // end namespace KMeans

#endif
