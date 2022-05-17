// See ChangeLog in header file for details

// implementation of the metric functions
#include <eigen2/Eigen/Array>

#include <stdlib.h>
#include "Metrics/Metrics.h"

using namespace std;
using namespace Eigen;

// Stats sets the avg, stdev, and sharpe variables to the calculated output
void Metrics::stats( double &avgRet, double &stdDev, double &sharpe, int &n,
	    MatrixXd &PnL) {

  // get the sum fist
  n = PnL.rows();
  avgRet = PnL.sum();

  // now divide by n to get the average
  avgRet /= n;

  // calculate spread from mean total
  stdDev = ((PnL.cwise() - avgRet).cwise().pow(2)).sum();

  // remember to divide by n-1 since this is sample standard deviation
  stdDev = sqrt(stdDev / ( n - 1 ));

  // and then calculate the sharpe from avg and stdev annualized
  sharpe = (avgRet * sqrt(261) ) / stdDev;

  return;

} // end stats
