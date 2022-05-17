// Author: Nathan Crosby
// Version: 0.0
// Date: 20210301
// ChangeLog:
//     0.0 -> initial creation

#ifndef METRICS_H
#define METRICS_H

namespace Metrics {
  // This is going to be a set of functions to help calculate metrics.
  void stats( double &, double &, double &, int &,
	      Eigen::MatrixXd &);
} // end namespace Metrics

#endif
