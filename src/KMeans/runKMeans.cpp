////////////////////////////////////////////////////////////////////////////////
// This is a quick run file to test the concept of finding similar memories   //
// in a fast manner from past examples.                                       //

#include <eigen2/Eigen/Array>

#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <iostream>
#include <sstream>
#include <fstream>

#include "./KMeans.h"

using namespace Eigen;
using namespace std;

int main(int argc, char* argv[]) {

  // Is it really better to cycle through the file one full time
  // instead of dynamically allocating the memory?
  ifstream inFile;
  inFile.open(argv[1]);
  string line;
  int M = 0;
  int N = 0;
  while(getline(inFile, line)) {
    stringstream thisLine(line);
    string value;
    if(M == 0)
      while(getline(thisLine, value, ','))
	++N;
    ++M;
  } // end while loop

  inFile.close();

  // Now reopen again and statically allocate the memory to contain it.
  inFile.open(argv[1]);

  vector<boost::posix_time::ptime> dates;
  MatrixXd history(M, N-2);

  int i = 0;
  while(getline(inFile, line)) {
    stringstream thisLine(line);
    string value;
    int j = 0;
    while(getline(thisLine, value, ',')) {
      if(j == 0)
	dates.push_back(boost::posix_time::time_from_string(value));
      else if(j > 1)
	history(i, j-2) = atof(value.c_str());
      ++j;
    } // end while loop storing data
    ++i;
  } // end file read while loop

  inFile.close();

  // outfile for results so I don't have to copy them from the terminal
  ofstream outFile, centerFile;
  
  // create the lagged dataset
  int lagStart = 10;
  int lagEnd = 100;
  int lagCol = 3;

  for(int lag = lagStart; lag <= lagEnd; lag++) {
    stringstream outFileName;
    outFileName << "../../inputs/Lag" << lag << ".csv";
    cout << outFileName.str() << endl;
    outFile.open(outFileName.str());
    stringstream centerFileName;
    centerFileName << "../../centers/Lag" << lag << ".csv";
    cout << centerFileName.str() << endl;
    centerFile.open(centerFileName.str());

    MatrixXd lags = MatrixXd::Zero(history.rows()-lag, lag);

    for(int i = 0; i < history.rows()-lag; i++) {
      for(int j = 0; j < lag; j++) {
	lags(i, j) = history(i + lag - j, lagCol) / history(i, lagCol) - 1.0;
      } // end column for
    } // end row for

    MatrixXd labels = MatrixXd::Zero(lags.rows(), 1);
    MatrixXd centers = KMeans::kmeans(lags, 10.0, labels);

    for(int i = 0; i < labels.rows(); i++) {
      outFile << dates[i+lag] << ", ";
      for(int k = 0; k < lags.cols(); k++)
	outFile << lags(i, k) << ", ";
      for(int j = 0; j < labels.cols(); j++) {
	outFile << labels(i, j);
	if(j != labels.cols() - 1)
	  outFile << ", ";
      }
      outFile << endl;
    }

    for(int i = 0; i < centers.rows(); i++) {
      for(int j = 0; j < centers.cols(); j++) {
	centerFile << centers(i, j);
	if(j != centers.cols() - 1)
	  centerFile << ", ";
      }
      centerFile << endl;
    }
  
    // close it up
    outFile.close();
    centerFile.close();
  } // end loop
  
  return 0;
  
} // end main function
