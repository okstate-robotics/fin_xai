// See ChangeLog in header file for details

// implementation of the kmeans functions
#include <eigen2/Eigen/Array>

#include <stdlib.h>
#include "./KMeans.h"

using namespace std;
using namespace Eigen;

// This function caclulates the euclidean distace between each observation
// and each center and assigns a label to each one based on the minimum. 
MatrixXd KMeans::calcDist(MatrixXd &input, MatrixXd &output) {
  // I can't think of a way to fully vectorize this right now
  int r = input.rows();
  int centers = output.rows();
  
  MatrixXd distances = MatrixXd::Zero(r, centers + 2);

  // iterate through all of the observations and calculated distances
  for(int i = 0; i < r; i++) {
    distances(i, 0) = i;
    double min = 1000000000000.0;
    // now iterate through all of the centers to calculate each distance
    for(int j = 0; j < centers; j++) {
      // euclidean distance
      distances(i, j + 1) = (input.row(i)-output.row(j)).norm();
      if(j == 0)
	min = distances(i, j + 1);
      // replace min and update label
      if(distances(i, j + 1) < min) {
	min = distances(i, j + 1);
	distances(i, centers + 1) = j;
      }
    } // end centers for loop
    
  } // end observations for loop

  return distances;

} // end calcCenters




// This just loops through and checks to see that all of the categories are
// equal to the previous loop
int KMeans::diffCat(MatrixXd &prev, MatrixXd &curr) {
  int r = prev.rows();
  int change = 0;
  
  for(int i = 0; i < r; i++) {
    if(prev(i, 0) != curr(i, 0))
      change++;
  } // end for loop

  return change;
} // end diffCat




// This function will move the the centers to their new location
void KMeans::calcCenterMean(MatrixXd &input, MatrixXd &cat, MatrixXd &output) {
  // get some basic info first
  int numCenters = output.rows();
  int dim = output.cols();
  int obs = input.rows();
  
  // use numCenters instead of worrying about unique categories, they go from
  // 0 to less than numCenters
  for(int cent = 0; cent < numCenters; cent++) {
    // create a new vector
    MatrixXd newCenter = MatrixXd::Zero(1, dim);
    int numRows = 0;
    for(int i = 0; i < obs; i++) {
      if(cat(i, 0) == cent) {
	newCenter = newCenter + input.row(i);
	numRows++;
      }
    } // end individual center for loop

    // now take the sum and divide by the num rows for the average
    if(numRows > 0)
      newCenter = newCenter / numRows;
    
    // now put the new row in the output matrix
    output.row(cent) = newCenter;
  } // end centers for loop

  // output already updated, so just return
  return;
  
} // end calcCenterMean




// This will be the main function that runs the kmeans algorithm
// I think just pass a pointer to the data set, and then I can take it from
// there. Maybe an optional input could be that the k is given so I don't
// have to search for it. 
MatrixXd KMeans::kmeans(MatrixXd &input, double inertiaThresh, MatrixXd &lab) { 
  // okay, so the outter most loop will figure out how many k is desired.
  // the main point is that as long as inertia is getting smaller, by a
  // significant* amount, then keep increasing k. Start with three I guess

  int M = input.rows();
  int N = input.cols();

  // This seeds the pseudo random number generator so that the runs are
  // different each time, otherwise rand() acts as if seeded with srand(1)
  unsigned seed = time(NULL);
  srand(seed);

  int centers = 3; // is there a good use case for starting at k = 1 ???
  double inertia = 0.0;
  double inertiaMax = 1000000000000.0;
  double prevInertia = inertiaMax;
  MatrixXd output; // create this out here so that it can be returned

  while(prevInertia - inertia > inertiaThresh) {
    // update prevInertia first
    if(inertia != 0.0 || prevInertia != inertiaMax)
      prevInertia = inertia;
    
    // create the output matrix and put in random points
    output = MatrixXd::Zero(centers, N);

    // now go through and load up some random centers to start with
    for(int i = 0; i < centers; i++) {
      int randRow = rand() % M;
      output.row(i) = input.row(randRow);
    } // end for loop

    int change = 1;
    int iter = 0;
    int minIter = 10;
    MatrixXd prev = VectorXd::Zero(M, 1);

    // now loop to until no points switch centers
    while(change > 0) {
      // distances is the euclidean distance from each center
      MatrixXd distances = calcDist(input, output);
      // Now see how many changed
      MatrixXd curr = distances.col(centers + 1);
      change = diffCat(prev, curr);

      // iterate here 
      iter++;

      // don't want to stop too soon
      if(iter < minIter && change == 0)
	change = 1;

      //cout << centers << ": change -> " << change << endl;

      // move centers if change is greater than one
      if(change > 0) {
	calcCenterMean(input, curr, output);
	prev = curr;
      } else {
	// calculate inertia here because about to break
	// this is just the sum of squares of distance to closest center
	inertia = 0.0;
	for(int i = 0; i < distances.rows(); i++)
	  inertia += distances(i, distances(i, centers + 1) + 1);

	// set the labels to curr here so that we can use them outside
	lab = curr;
      }

    } // end while for search inside single k

    if(prevInertia - inertia <= inertiaThresh)
      cout << "seed: " << seed << " -- " << N << " -- " << centers
	   << ": Inertia -> " << inertia << endl;

    centers++;

  } // end while search for optimal k

  return output; 
} // end kmeans
