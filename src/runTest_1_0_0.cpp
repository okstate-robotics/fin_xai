// Author: Nathan Crosby
// Version: 1.0.0
// Date: 20210301
// ChangeLog:
//     0.0 -> initial creation
//     1.0.0 -> main difference is changing to use Kelley Criterion instead
//              of the threshBuy and threshSell

////////////////////////////////////////////////////////////////////////////////
// This is a quick run file to test the concept of finding similar memories   //
// in a fast manner from past examples.                                       //

#include <eigen2/Eigen/Array>

#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

#include "HyperParam/HyperParam.h"
#include "Metrics/Metrics.h"
#include "KMeans/KMeans.h"

using namespace Eigen;
using namespace std;





// struct to keep track of all of the parameters passed to the calculation
struct params {
  int M, startTime, i, pass;
  double feePerSide, multiple;

  // hyperparameters
  int lookback, topNum, outNum;
  double kellyMult;

  // Add category MatrixXds here since boost bind only accepts 9 arguments
  MatrixXd centers, groupings;
};





// Eigen does not have a good print function that I know of
void printMat(MatrixXd &mat, ostream &output /*= std::cout*/) {
  // find the number of rows and columns
  int M = mat.rows();
  int N = mat.cols();
  // cycle through all of the elements
  for( int i = 0; i < M; ++i ) {
    for( int j = 0; j < N; ++j ) {
      // Then print them out
      if( j == N-1 )
        output << mat(i, j);
      // add comma if you are not at the end
      else
        output << mat(i, j) << ",";
    }
    // New line every line in matrix
    output << endl;
  }
  return;
}




// find the end of the week
int findWeekend(vector<boost::posix_time::ptime> &datesPer, int pos) {

  // Since we have the pos, find the nearest weekend
  while(pos < datesPer.size() - 1 &&
	datesPer[pos].date().day_of_week() <
	(datesPer[pos+1]).date().day_of_week()) {
    pos++;
  } // end while

  return pos;
} // end findWeekend





// convenience function to return index number of date in file
int findDate(vector<boost::posix_time::ptime> &datesPer,
	     boost::posix_time::ptime date, int weekend) {
  // I want to return the position, but everything is in chronological
  // order, so use a binary search then find the date 
  int left = 0;
  int right = datesPer.size() - 1;
    
  int closest = -1;
  
  while(left <= right) {
    int mid = (right + left) / 2;
    if(date == datesPer[mid])
      right = 0; // this should break the loop
    else if(date < datesPer[mid]) 
      right = mid - 1;
    else 
      left = mid + 1;

    if(left > right)
      closest = mid;
  } // end while

  if(weekend != 0) 
    closest = findWeekend(datesPer, closest);

  return closest;
} // end findDate




// find center to use to look up value
int findCenter(MatrixXd &curr, MatrixXd &centers) {
  // Grab the rows and columns 
  int M = centers.rows();
  int N = centers.cols();

  // Double check that the sizes are the same
  if(curr.rows() != N)
    return -1;

  double lowMSE = 1000000.0;
  int bestFit = -1;
  for(int i = 0; i < M; i++) {
    // each row is a different center, so loop through and keep the best
    double MSE = 0.0;
    for(int j = 0; j < N; j++) {
      MSE += pow(curr(j, 0) - centers(i, (N - 1) - j), 2.0);
    } // end MSE for loop
    if(MSE < lowMSE) {
      lowMSE = MSE;
      bestFit = i;
    } // end best check
  } // end centers for loop

  /*
  cout << "Current: " << endl;

  printMat(curr, cout);

  cout << endl << "Centers: " << endl;

  printMat(centers, cout);

  cout << "Best fit is: " << bestFit << endl;
  cout << "Low MSE: " << lowMSE << endl;
  */
  
  return bestFit;
} // end findCenter





// Quick function to calculate and return the cumulative sum of series
MatrixXd cumSum(MatrixXd &mat) {
  // Grab the rows and columns 
  int M = mat.rows();
  int N = mat.cols();

  // Set the first row to the first row of the input
  MatrixXd output = MatrixXd::Zero(M, N);
  output.row(0) = mat.row(0);

  // Then just loop through and sum
  for(int j = 1; j < M; j++)
    output.row(j) = output.row(j-1) + mat.row(j);

  // Finally just return the output
  return output;
 
} // end cumSum





// This is probably not the best test. Correlation or Covariance
// might be a slight upgrade, but cointegration would probably
// be best. 
double matchScore(MatrixXd &curr, MatrixXd &comp) {
  // Double check that the sizes are the same
  if(curr.rows() != comp.rows() || curr.cols() != comp.cols())
    return -1.0;

  // Grab the rows and columns 
  int M = curr.rows();
  int N = curr.cols();

  // 1 - (sum{abs[(curr - comp) / curr]} / M)
  double out = 1.0 - (((curr - comp) / curr(M-1, 0)).cwise().abs().sum() / M);

  return out;

} // end matchScore function





// used to sort the genetic pool after it is done running
struct sortGen {
    bool operator()(const pair<double, map<string, Hype::HyperParam> > &left,
		    const pair<double, map<string, Hype::HyperParam> > &right) {
        return left.first < right.first;
    }
}; // end sortGen






// This is the asyncronous function that will be called to run each test
void testProcess(params par, MatrixXd &histPer,
		 vector<boost::posix_time::ptime> &datesPer,
		 MatrixXd &history, double &topPrecision,
		 vector< pair< double, map<string, Hype::HyperParam> > >
		 &geneticPool,
		 ofstream &outfile, double maxPrecision) {

  double position = 0;
  int perfPos = 0;

  int M = par.M;
  if(M < datesPer.size() && par.pass < 0)
    M++;
  int startTime = par.startTime;
  double tarRet = 50.0;

  MatrixXd Metrics = MatrixXd::Zero(M-startTime, 1);
  MatrixXd PnL = MatrixXd::Zero(M-startTime, 1);
  MatrixXd PnLPerf = MatrixXd::Zero(M-startTime, 1);
  MatrixXd PnLBuyHold = MatrixXd::Zero(M-startTime, 1);

  boost::posix_time::ptime mst1;
  mst1 = boost::posix_time::microsec_clock::local_time();

  // These two are used alot so convenience variables here
  int lb = par.lookback;
  int tn = par.topNum;
  int gpNum = par.i;
  int pass = par.pass;

  if(pass < 0)
    cout << "running -1 pass in testProcess" << endl;

  /*
  cout << "gpNum: " << gpNum << "  start: " << startTime << "  end: "
       << M << "  Lookback: " << lb << "  topNum: " << tn << "  outNum: "
       << par.outNum << "  kellyMult: " << par.kellyMult << endl;
  */
  
  // Double check that the problem doesn't persist
  if(par.outNum > lb) {
    cout << "\n\n\nProblem with parameters\n\n\n" << endl;
    exit(-1);
  }

  // Outter day loop
  for(int day = startTime; day < M; day++) {
    MatrixXd currList = histPer.block(day-lb, 3, lb, 1);
    MatrixXd currListCum = cumSum(currList);

    ////////////////////////////////////////////////////////////////////////////
    // For each day, find out the best center for the current lookback,
    // that way we can eliminate the other searches from the loop

    int center = findCenter(currListCum, par.centers);
    int groupRows = par.groupings.rows();

    //cout << par.groupings.rows() << ", " << par.groupings.cols() << endl;
    //cout << "lb: " << lb << "\tstartTime: " << startTime << endl;

    // There are exactly startTime - lb entries in the groupings matrix, so
    // the check should then be on center == par.groupings(i, 0)

    ////////////////////////////////////////////////////////////////////////////
    
    MatrixXd best = MatrixXd::Zero(tn, 2);
    for(int i = 0; i < tn; i++)
      best(i, 1) = -1000000.0;

    // For a start, loop through to see if a really close match for the current
    // can be found at all. Then move to prediction
    for(int i = day - lb, j = 0; i >= lb; i--, j++) {
      if(i >= groupRows || center == par.groupings(i, 0)) {
	MatrixXd compList = histPer.block(i - lb, 3, lb, 1);
	MatrixXd compListCum = cumSum(compList);

	// Primary function that calculates the match score between
	// current and each "memory" in the past
	double currMatch = matchScore(currListCum, compListCum);
	if(currMatch == -1.0)
	  cout << "An error has occurred, -1 was returned on day: " << day
	       << " and i: " << i << endl;

	// now swap around if necessary
	int bestCk = tn-1;

	// first check to see if it even made the top list
	if(currMatch > best(bestCk, 1)) {
	  // and put it in the last position if it did.
	  best(bestCk, 0) = i;
	  best(bestCk, 1) = currMatch;
	  bestCk--;
	  double temp0, temp1;
	  temp0 = temp1 = 0.0;
	  
	  // Now iterate though and see if it should move up the list. 
	  while(bestCk >= 0 && currMatch > best(bestCk, 1)) {
	    // do the standard swap if it should
	    temp0 = best(bestCk, 0);
	    temp1 = best(bestCk, 1);
	    best(bestCk, 0) = best(bestCk + 1, 0);
	    best(bestCk, 1) = best(bestCk + 1, 1);
	    best(bestCk + 1, 0) = temp0;
	    best(bestCk + 1, 1) = temp1;
	    bestCk--;
	  } // end swap while loop
	} // end if statement for currMatch insert
      } // end if check or not
    } // end for loop

    // No longer used??? Remove compIn if so
    //MatrixXd compIn = MatrixXd::Zero(lb, tn + 1);
    //compIn.col(0) = currListCum;
    
    // Now pull all of the histories for the best list
    MatrixXd compOut = MatrixXd::Zero(lb, tn);

    for(int i = 0; i < tn; i++) {
      //currList = histPer.block(best(i, 0) - lb, 3, lb, 1);
      //currListCum = cumSum(currList);
      //compIn.col(i+1) = currListCum;
      MatrixXd compList = histPer.block(best(i, 0) + 1, 3, lb, 1);
      MatrixXd compListCum = cumSum(compList);
      compOut.col(i) = compListCum;
    }
  
    // Let's create a metric now, so that a decision can be made
    MatrixXd weights = best.col(1) / best.col(1).sum();
    MatrixXd weightedFuture = compOut * weights;
    
    //double met = (double) weightedFuture(par.outNum-1, 0);

    ////////////////////////////////////////////////////////////////////////////
    // **** Switching this to use Kelly Criterion ****
    // Instead of guessing what the buy and sell levels should be we will let
    // the probabilities and expected returns dictate the investment level.
    // Still use the outNum variable to decide how far out in the future to look
    // but instead of just taking the weighted average of everything and looking
    // to see if that future expected return is above of below a certain number
    // I am now switching to f* = p - q/odds; where p is pos/total,  q = 1 - p,
    // and odds is E[r]pos / E[r]neg

    double p = 0.0, exPos = 0.0, posSum = 0.0, exNeg = 0.0, negSum = 0.0;

    for(int e = 0; e < tn; e++) {
      double check = compOut(par.outNum-1, e);
      if(check >= 0.0) {
	p += 1.0;
	exPos += check * weights(e, 0);
	posSum += weights(e, 0);
      } else {
	exNeg -= check * weights(e, 0); // subtract to make positive number
	negSum += weights(e, 0);
      }
    }

    // Normalize
    p /= tn;
    if(posSum > 0.0)
      exPos /= posSum;
    if(negSum > 0.0)
      exNeg /= negSum;

    // then use to calculate fStar
    double fStar = 1.0;
    double q = 0.0;
    if(p < 1.0 && p > 0.0) {
      q = 1.0 - p;
      fStar = p - (q/(exPos/exNeg));
    } else if(p == 0.0) {
      q = 1.0;
      fStar = -1.0;
    }
    ////////////////////////////////////////////////////////////////////////////

    // Now that the new fStar metric is implemented, incorporate that into
    // the positions. Limit to 100% positive or negative.
    double met = fStar;
    if(met > 1.0)
      met = 1.0;
    else if(met < -1.0)
      met = -1.0;
    Metrics(day - startTime, 0) = met;
    
    // track fees and keep track of position (allow partial position now)
    double newPos = met * par.kellyMult;
    double trades = abs(position - newPos);
    double fees = -1 * trades * par.feePerSide;
    position = newPos;
    
    // get current and next value to calculate P&L
    double curr = history(day, 3);
    double next = curr;
    if(day+1 <= M)
      next = history(day+1, 3);

    // Perfect information fees depend on if perfPos will change
    double perfFees = 0.0;
    if(next > curr && perfPos < 1) {
      perfFees = -2 * par.feePerSide;
      perfPos = 1;
    } else if(next < curr && perfPos > -1) {
      perfFees = -2 * par.feePerSide;
      perfPos = -1;
    }
    
    PnL(day - startTime, 0) = (next-curr)*position*par.multiple + fees;
    PnLPerf(day - startTime, 0) = abs(next-curr)*par.multiple + perfFees;
    PnLBuyHold(day - startTime, 0) = (next-curr)*par.multiple; 

    ////////////////////////////////////////////////////////////////////////////
    // This is only if it is flagged as test instead of train. 
    if(pass < 0) {
      outfile << datesPer[day] << "," << day << "," << lb << "," << tn << ","
	      << par.outNum << "," << par.kellyMult << ",\"{";

      // loop to print out all of the top matches
      for(int i = 0; i < tn; i++) {
	if(i > 0)
	  outfile << ",";
	outfile << best(i, 0);
      } // end best match loop

      outfile << "}\",\"{";

      int nonZeroWeights = 0;
      double weightSum = 0.0;
      // loop to print out all of the weights for those matches. 
      for(int i = 0; i < tn; i++) {
	if(i > 0)
	  outfile << ",";
	outfile << weights(i, 0);

	if(weights(i, 0) != 0.0) {
	  nonZeroWeights++;
	  weightSum += weights(i, 0);
	}
      } // end weights loop
      
      outfile << "}\",\"{";

      MatrixXd squaredDiff = compOut;
      // put in weighted average future returns
      for(int i = 0; i < lb; i++) {
	if(i > 0)
	  outfile << ",";
	outfile << weightedFuture(i, 0);

	for(int j = 0; j < tn; j++) 
	  squaredDiff(i, j) = pow(squaredDiff(i,j) - weightedFuture(i,0), 2.0);
      } // end weighted average loop

      outfile << "}\",\"{";

      MatrixXd weightedStdev = squaredDiff * weights;
      double denominator = ((double)(nonZeroWeights - 1) /
			    (double)nonZeroWeights) * weightSum;

      weightedStdev = (weightedStdev / denominator).cwise().pow(0.5);
      
      // put in weighted standard deviation of future returns
      for(int i = 0; i < lb; i++) {
	if(i > 0)
	  outfile << ",";
	outfile << weightedStdev(i, 0);
      } // end weighted standard deviation loop

      outfile << "}\"," << p << ", " << q << ", " << exPos << ", " << exNeg
	      << ", " << position << ", " << PnL(day - startTime, 0) << ","
	      << PnLPerf(day - startTime, 0) << ","
	      << PnLBuyHold(day - startTime, 0)
	      << ",\"From the " << tn << " relevant examples found in group "
	      << center << ", there were was a " << (round(p * 10000) / 100)
	      << "% probability of a positive return with an expected gain of "
	      << (round(exPos * 10000) / 100) << "%. A loss of "
	      << (round(exNeg * 10000) / 100)
	      << "% was expected if a negative return was experienced. "
	      << " With these factors taken into consideration, a position of "
	      << position << " was entered.\"" << endl;
    }
    ////////////////////////////////////////////////////////////////////////////

  } // end outter loop

  boost::posix_time::ptime mst2;
  mst2 = boost::posix_time::microsec_clock::local_time();
  boost::posix_time::time_duration msdiff = mst2 - mst1;

  if(pass >= 0) {
    // sum up all of the PnLs
    MatrixXd PnLs = MatrixXd::Zero(M - startTime, 3);
    PnLs.col(0) = cumSum(PnL);
    PnLs.col(1) = cumSum(PnLPerf);
    PnLs.col(2) = cumSum(PnLBuyHold);

    // take the percentage of perfect as precision
    // old way
    //double prec = PnLs(M-startTime-1, 0)/PnLs(M-startTime-1, 1);

    ////////////////////////////////////////////////////////////////////////////
    // ********** Updated precision calculation ************
    
    // first need the metrics from the stat function
    double avgRet, stdDev, sharpe;
    avgRet = stdDev = sharpe = 0.0;
    int n = 0;
    
    Metrics::stats(avgRet, stdDev, sharpe, n, PnL);
    
    //cout << "Ret: " << avgRet << "\tstdev: " << stdDev << "\tsharpe: "
    //     << sharpe << endl;

    double perfRet, perfDev, perfSharpe;
    perfRet = perfDev = perfSharpe = 0.0;
    int perfN = 0;

    Metrics::stats(perfRet, perfDev, perfSharpe, perfN, PnLPerf);

    //cout << "perf ret: " << perfRet << "\tperf stdev: " << perfDev
    //     << "\tperf sharpe: " << perfSharpe << endl;
  
    double prec = -10000.0;
    // Just use average first
    prec = avgRet / perfRet;
    // How about sharpe now
    /*
    prec = sharpe / perfSharpe;
    */
    // This is scaled sharpe
    /*
    if(stdDev > 0) 
      prec = (abs(sharpe) * (avgRet/(abs(avgRet)+tarRet))) /
	(abs(perfSharpe) * (perfRet/(abs(perfRet)+tarRet)));
    */
    // use match score to approximate MSE
    /*
    MatrixXd perfCum = PnLs.col(1);
    MatrixXd runCum = PnLs.col(0);
    prec = matchScore(perfCum, runCum);
    */
    //cout << "precision: " << prec << endl;

    ////////////////////////////////////////////////////////////////////////////
  
    // if it is the best, replace the best. (Sorting happens after full loop so
    // worry about locking shared variable later if you care.)
    if(prec > topPrecision)
      topPrecision = prec;
    
    geneticPool[gpNum].first = prec;
    
    outfile << "Pass: " << pass << " -> gpNum: " << gpNum << "\tHyper: " << lb
	    << ", " << tn << ", " << par.outNum << " \tkm: " << par.kellyMult
	    << "\tavgRet: " << avgRet << "\tsharpe: " << sharpe << "\tperfRet: "
	    << perfRet << "\tperfSharpe: " << perfSharpe << "\t%Perf: "
	    << prec << "  %BuyHold: "
	    << PnLs(M-startTime-1, 0)/PnLs(M-startTime-1, 2) << "\tLoopTop: "
	    << topPrecision << "  Max: " << maxPrecision << "\tTime: " << msdiff
	    << endl;
  } // end if
  
} // end testProcess





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
  ofstream outfile, outResults;
  outfile.open("./outfile.csv");
  outResults.open("./outResults.csv");

  // convert to percentages first
  MatrixXd histPer = (history.block(1, 0, M-1, N-2).cwise() /
		      history.block(0, 0, M-1, N-2)).cwise() - 1.0;
  vector<boost::posix_time::ptime> datesPer =
               vector<boost::posix_time::ptime>(dates.begin() + 1, dates.end());

  // As an initial test, start 90% of the way through and predict last 10%
  //int startTime = (int) (0.997275 * M);
  //////////////////////////////////////////////////////////////////////////////
  // Switching this up to look for the start time in the file and start from
  // there. Will need a find next week place function that facilitates going
  // from one test week to the next
  boost::posix_time::ptime startDate =
    boost::posix_time::time_from_string("2019-10-04 15:00:00"); // 10/4/19
  int endTime = findDate(datesPer, startDate, 1);
  int trainLen = 5040;  // 504 -> 2 years, 5040 -> 20 years
  int startTime = 0;
  if(endTime > trainLen)
    startTime = endTime - trainLen;
  int testWeeks = 65; // 13
  //////////////////////////////////////////////////////////////////////////////

  // ***** FEES *****
  // $2.72 for CL on TS, $2.4 for ES,
  // $2.72 for GC, $2.82 for EC
  double feePerSide = 2.4;
  double multiple = 50.0;

  // first need a vector to hold keys and a map to hold key, hyperparam pairs
  vector<string> keys;
  map<string, Hype::HyperParam> hyperParams;

  // This seeds the pseudo random number generator so that the runs are
  // different each time, otherwise rand() acts as if seeded with srand(1)
  unsigned seed = time(NULL);
  outfile << "seed: " << seed << endl; // print out so can be replicated later
  srand(seed);

  params par;

  int numSteps = 10; // change back to 10

  // put in hyperparameters
  Hype::HyperParam lookback(30, 30, 1, numSteps); // window size 10 -> 100
  hyperParams.insert(pair<string, Hype::HyperParam>("lookback", lookback));
  keys.push_back("lookback");

  Hype::HyperParam topNum(2, 30, 1, numSteps); // number of examples 2 -> 30
  hyperParams.insert(pair<string, Hype::HyperParam>("topNum", topNum));
  keys.push_back("topNum");

  Hype::HyperParam outNum(1, 100, 1, numSteps); // how far in future  1 -> 100
  hyperParams.insert(pair<string, Hype::HyperParam>("outNum", outNum));
  keys.push_back("outNum");

  Hype::HyperParam kellyMult(0, 20, 10, numSteps); // 0 -> 2 by 0.1
  hyperParams.insert(pair<string, Hype::HyperParam>("kellyMult", kellyMult));
  keys.push_back("kellyMult");

  // Find out how many threads are available to use
  int maxThreads = 7;
  unsigned int concurrentThreads = boost::thread::hardware_concurrency() - 1;
  cout << "Additional concurent threads supported: "
       << concurrentThreads << endl;
  if(concurrentThreads > maxThreads) {
    concurrentThreads = maxThreads;
    cout << "Limiting to " << maxThreads << " additional threads" << endl;
  } // end max thread check

  int multiplier = 10; // change back to 10
  int poolSize = concurrentThreads * multiplier * numSteps; // Seems about right
  int maxListSize = concurrentThreads * multiplier;

  // create the vector to hold the multiple lists of hyperparams and results
  vector< pair< double, map<string, Hype::HyperParam> > > geneticPool;
  for(int i = 0; i < poolSize; i++) {
    geneticPool.push_back(make_pair(-10000.0, hyperParams));
  }
  vector< pair< double, map<string, Hype::HyperParam> > > maxList;
  for(int i = 0; i < maxListSize; i++) {
    maxList.push_back(make_pair(-10000.0, hyperParams));
  }

  /////////////////////////////////////////////////////
  /* This is where the date loop will start.         */
  /////////////////////////////////////////////////////
  while(endTime < (M - 1)) {
    cout << "Training from: { " << datesPer[startTime] << ", "
	 << datesPer[endTime] << " } " << endl;
    outfile << "Training from: { " << datesPer[startTime] << ", "
	    << datesPer[endTime] << " } " << endl;

    ////////////////////////////////////////////////////////////////////////////
    // ************** Generate KMeans Groups ******************               //
    // Each time we step forward in time we need to regroup things. No need to//
    // store these values, as the group numbers are meaningless. Only the     //
    // decision tree interpretations will matter in the end (They are just    //
    // used as a way of describing how the groups are segmented).             //
    int lagStart = geneticPool[0].second["lookback"].getMin();
    int lagEnd = geneticPool[0].second["lookback"].getMax();
    int lagCol = 3;

    // Create a vector to hold all of the groupings out here
    // histPer is only 
    vector<MatrixXd> groupings;
    // centers needs to be a list of matrices with the centers
    vector<MatrixXd> lagCenters; 

    for(int lag = lagStart; lag <= lagEnd; lag++) {
      // for each lag, I need a grouping. 
      MatrixXd lags = MatrixXd::Zero(startTime - lag, lag);

      // don't want to go past start time or that would be looking into
      // the future
      for(int i = 0; i < startTime - lag; i++) {
	for(int j = 0; j < lag; j++) {
	  lags(i, j) = history(i + lag - j, lagCol) / history(i, lagCol) - 1.0;
	}
      }

      // Run the kmeans methods and save the output 
      MatrixXd labels = MatrixXd::Zero(lags.rows(), 1);
      MatrixXd centers = KMeans::kmeans(lags, 10.0, labels);
      groupings.push_back(labels);
      lagCenters.push_back(centers);

      // startTime is based off of datesPer, but the lags going into the
      // kmeans use history. Do I need a +1 somewhere???? Test this out. 

    } // end lag for loop

    //cout << "Size of groupings: " << groupings.size() << endl;
    //cout << "Size of centers: " << lagCenters.size() << endl;

    //cout << "first grouping: " << endl;
    //printMat(groupings[0], cout);

    //cout << "first centers: " << endl;
    //printMat(lagCenters[0], cout);
        
    ////////////////////////////////////////////////////////////////////////////

    for(int i = 0; i < maxListSize; i++) {
      // thought about a loop, but two of the five variables have exceptions
      // so just enumerate it here. 
      maxList[i].second["lookback"].nextRand();
      maxList[i].second["topNum"].nextRand();
      maxList[i].second["outNum"].nextRand(
	maxList[i].second["lookback"].getValue());
      maxList[i].second["kellyMult"].nextRand();

      // now set the set
      for(int j = 0; j < keys.size(); j++) {
	maxList[i].second[keys[j]].setSet(i);
      }
      // don't forget this
      maxList[i].first = -10000.0;
    } // end geneticPool initialization

    // now load up the genetic pool
    for(int i = 0; i < poolSize; i++) {
      geneticPool[i] = maxList[i % maxListSize];
    }

    double maxPrecision = -10000.0;
    double convergence = 0.0005;
    int pass = 0;
    int consecNoProg = 0; // not getting enough diversity
    int limitNoProg = 10; // so make it 

    // Loop until convergence is met
    while(consecNoProg < limitNoProg) {
      cout << "Loop: " << pass << endl;
      // As long as it is not the first pass, then update pool
      if(pass != 0) {
	// replicate in the geneticPool
	for(int i = 0; i < poolSize; i++) {
	  // couple of convenience variables
	  int repNum = i / maxListSize; // int division
	  int set = i % maxListSize; // remainder, already set when initialized

	  geneticPool[i] = maxList[set];

	  // calculate direction and magnitude
	  int stepToTake = (repNum / 2) + 1;
	  if(repNum % 2 != 0)
	    stepToTake *= -1; // flip sign if odd number remainder

	  // set direction
	  for(int k = 0; k < keys.size(); k++) {
	    geneticPool[i].second[keys[k]].setDirection(stepToTake);
	  }
	} // end replicate maxList/geneticPool setup

	// Now update the genetic pool based on keyUpdate and step size
	for(int i = 0; i < poolSize; i++) {
	  outfile << i << " --> old: "
		  << geneticPool[i].second["lookback"].getValue() << ", "
		  << geneticPool[i].second["topNum"].getValue() << ", "
		  << geneticPool[i].second["outNum"].getValue() << ", "
		  << geneticPool[i].second["kellyMult"].getValue();

	  int upKey = geneticPool[i].second[keys[0]].getKeyUpdate();
	  geneticPool[i].second[keys[upKey]].nextStep();

	  int thisStep = geneticPool[i].second[keys[upKey]].getStep();
	  int thisDir = geneticPool[i].second[keys[upKey]].getDirection();
	  outfile << " ==> step: " << thisStep << " direction: " << thisDir
		  << " ==> new: (" << keys[upKey] << ") -> "
		  << geneticPool[i].second[keys[upKey]].getValue();

	  // check the exceptions that might invalidate the results
	  if(geneticPool[i].second["outNum"].getValue() >
	     geneticPool[i].second["lookback"].getValue()) {
	    geneticPool[i].second["outNum"].setValue(
	      geneticPool[i].second["lookback"].getValue());
	    outfile << " { outNum violation } newnew: "
		    << geneticPool[i].second["outNum"].getValue();
	  }

	  outfile << endl;

	  // increment turns (which also reduces the step)
	  geneticPool[i].second[keys[upKey]].incrementTurns();

	} // end genetic pool restructure loop

	cout << "Gene pool restructured" << endl;
      } // end pool restructure



      // create the io_service to post tasks to
      boost::asio::io_service io_service;
      double loopTopPrecision = -10000.0;

      // now loop through every set of parameters and calculate the outcome
      for(int i = 0; i < poolSize; i++) {
	// put all of the parameters in the struct to pass to the process
	par.M = endTime;
	par.startTime = startTime;
	par.feePerSide = feePerSide;
	par.multiple = multiple;
	par.lookback = (int)geneticPool[i].second["lookback"].getValue();
	par.topNum = (int)geneticPool[i].second["topNum"].getValue();
	par.outNum = (int)geneticPool[i].second["outNum"].getValue();
	par.kellyMult = geneticPool[i].second["kellyMult"].getValue();
	par.i = i;
	par.pass = pass;

	// put centers and groupings in par since there are not enough
	// available parameters for the boost bind function
	int lb = par.lookback;

	par.centers = boost::ref(lagCenters[lb-lagStart]);
	par.groupings = boost::ref(groupings[lb-lagStart]);

	// post those tasks to the service
	io_service.post(boost::bind(&testProcess, par, boost::ref(histPer),
				    boost::ref(datesPer),
				    boost::ref(history),
				    boost::ref(loopTopPrecision),
				    boost::ref(geneticPool),
				    boost::ref(outfile), maxPrecision));

      } // end run loop

      // now create the threads and connect them to the io_service
      boost::thread_group threads;
      
      for(size_t i = 0; i < concurrentThreads; ++i) {
	threads.create_thread(boost::bind(&boost::asio::io_service::run,
					  &io_service));
      } // end thread creation

      threads.join_all();

      // **** UPDATE: Don't need this anymore ****
      // sort the genetic pool based on precision
      //sort(geneticPool.begin(), geneticPool.end(), sortGen());
      // Instead, just decide if it should replace the one in maxList
      for(int gp = 0; gp < poolSize; gp++) {
	outfile << gp;
	double newRes = geneticPool[gp].first;
	int thisSet = geneticPool[gp].second[keys[0]].getSet();
	double oldRes = maxList[thisSet].first;
	outfile << ": check greater => newRes: " << newRes << " oldRes: "
		<< oldRes << " thisSet: " << thisSet << endl;
	if(newRes > oldRes) 
	  maxList[thisSet] = geneticPool[gp];
	else if(newRes == oldRes) {
	  // now we have to see if the updated value is closer to the middle
	  int upKey = geneticPool[gp].second[keys[0]].getKeyUpdate();
	  outfile << "upKey: " << upKey << endl;
	  if(upKey > -1) {
	    int thisMax = geneticPool[gp].second[keys[upKey]].getMax();
	    int thisMin = geneticPool[gp].second[keys[upKey]].getMin();
	    int thisDivisor = geneticPool[gp].second[keys[upKey]].getDivisor();
	    double thisMid = ((double)(thisMin + (thisMax - thisMin) / 2)) /
	      (double)thisDivisor;
	    if(abs(geneticPool[gp].second[keys[upKey]].getValue() - thisMid) <
	       abs(maxList[thisSet].second[keys[upKey]].getValue() - thisMid))
	      maxList[thisSet] = geneticPool[gp];
	    else {
	      // turns is incremented in geneticPool[gp], but if we don't use
	      // it then still want to set turns in existing maxList item
	      // ******* QUESTION: SHOULD THERE ALSO BE A REDUCE STEP HERE?????
	      // (or at least a set step to the gp step????)
	      // Maybe we do not want to reduce the step if the gp run is not
	      // ever accpeted in the run!
	      // On second thought, yeah, reduce it since nothing good was
	      // found at that level.
	      int thisTurns = geneticPool[gp].second[keys[upKey]].getTurns();
	      int thisStep = geneticPool[gp].second[keys[upKey]].getStep();
	      maxList[thisSet].second[keys[upKey]].setTurns(thisTurns);
	      maxList[thisSet].second[keys[upKey]].setStep(thisStep);
	    }
	  }
	} else if(geneticPool[gp].second[keys[0]].getKeyUpdate() > -1) {
	  // turns is incremented in geneticPool[gp], but if we don't use
	  // it then still want to set turns in existing maxList item
	  // ******* QUESTION: SHOULD THERE ALSO BE A REDUCE STEP HERE?????
	  // (or at least a set step to the gp step????)
	  int upKey = geneticPool[gp].second[keys[0]].getKeyUpdate();
	  int thisTurns = geneticPool[gp].second[keys[upKey]].getTurns();
	  int thisStep = geneticPool[gp].second[keys[upKey]].getStep();
	  maxList[thisSet].second[keys[upKey]].setTurns(thisTurns);
	  maxList[thisSet].second[keys[upKey]].setStep(thisStep);
	}
      }

      // update minTurns
      for(int i = 0; i < maxListSize; i++) {
	int minT = maxList[i].second[keys[0]].getTurns();
	for(int t = 1; t < keys.size(); t++) {
	  if(maxList[i].second[keys[t]].getTurns() < minT)
	    minT = maxList[i].second[keys[t]].getTurns();
	}
	if(maxList[i].second[keys[0]].getMinTurns() != minT) {
	  for(int t = 0; t < keys.size(); t++)
	    maxList[i].second[keys[t]].setMinTurns(minT);
	}
      }

      // pick a random key
      for(int i = 0; i < maxListSize; i++) {
	int hypePick = rand() % keys.size();

	outfile << i << " --> original: " << hypePick;
	// make sure it is one that is on the min turns level
	while(maxList[i].second[keys[hypePick]].getTurns() !=
	      maxList[i].second[keys[hypePick]].getMinTurns()) {
	  int currTurns = maxList[i].second[keys[hypePick]].getTurns();
	  int currMinTurns = maxList[i].second[keys[hypePick]].getMinTurns();
	  hypePick = (hypePick + 1) % keys.size();
	  outfile << " { turns: " << currTurns << ", minTurns: " << currMinTurns
		  << " => next: " << hypePick << " }";
	}
	outfile << endl;
	// now set all of the key updates
	for(int k = 0; k < keys.size(); k++) 
	  maxList[i].second[keys[k]].setKeyUpdate(hypePick);
      }

      // find max precision
      double newMaxPrecision = maxList[0].first;
      for(int ml = 1; ml < maxListSize; ml++) {
	if(maxList[ml].first > newMaxPrecision)
	  newMaxPrecision = maxList[ml].first;
      }

      cout << "Sorted, testing loop again. New Max: "
	   << newMaxPrecision << endl;

      // check to see if progress is made. 
      if((newMaxPrecision - maxPrecision) < convergence)
	consecNoProg++;
      else
	consecNoProg = 0;

      maxPrecision = newMaxPrecision;

      pass++;
    
    } // end of convergence loop

    outfile << "Top Genetic Results:" << endl;
    for(int i = 0; i < maxListSize; i++) {
      int lb = (int)maxList[i].second["lookback"].getValue();
      int tn = (int)maxList[i].second["topNum"].getValue();
      int on = (int)maxList[i].second["outNum"].getValue();
      double km = maxList[i].second["kellyMult"].getValue();
      outfile << i << ": " << maxList[i].first << "\tHyper: " << lb
	      << ", " << tn << ", " << on << " \tkm: " << km << endl;
    }// end output for loop

    //////////////////////////////////////////////////////////
    /* Now run the next week to put values in the db.       */
    //////////////////////////////////////////////////////////

    // if we end training on a Friday, then start up following Monday
    // (If it is a valid trading day, of course)
    startTime = endTime + 1;

    if(startTime < (M - 1)) {
      cout << "Running -1 pass at start time: " << startTime << endl;
      for(int tw = 0; tw < testWeeks; tw++) {
	if(endTime < (M - 1)) // check this again, because we are incrementing
	  endTime++;
	endTime = findWeekend(datesPer, endTime); // this is safe
      } // end for loop
      cout << "New end time: " << endTime << endl;
      
      par.M = endTime;
      par.startTime = startTime;
      par.feePerSide = feePerSide;
      par.multiple = multiple;

      // from top genetic resutls
      int best = 0;
      double bestRes = maxList[0].first;
      for(int i = 0; i < maxListSize; i++) {
	if(maxList[i].first > bestRes) {
	  best = i;
	  bestRes = maxList[i].first;
	}
      }
      
      par.lookback = (int)maxList[best].second["lookback"].getValue();
      par.topNum = (int)maxList[best].second["topNum"].getValue();
      par.outNum = (int)maxList[best].second["outNum"].getValue();
      par.kellyMult = maxList[best].second["kellyMult"].getValue();
      par.i = 1;
      par.pass = -1;

      // put centers and groupings in par since there are not enough
      // available parameters for the boost bind function
      int lb = par.lookback;
      int lagStart = maxList[best].second["lookback"].getMin();

      par.centers = boost::ref(lagCenters[lb-lagStart]);
      par.groupings = boost::ref(groupings[lb-lagStart]);
      
      double topPrecision = 0.0;

      testProcess(par, histPer, datesPer, history, topPrecision, geneticPool,
		  outResults, maxPrecision);

      // set this back for next trainging process
      startTime = endTime - trainLen;
    } else
      endTime = startTime + 1;
    // end test process
    
  }
  //////////////////////////////////////////////////////////
  /* This should be the end of the date loop              */
  //////////////////////////////////////////////////////////
  
  // close it up
  outfile.close();
  outResults.close();
  
  return 0;
  
} // end main function
