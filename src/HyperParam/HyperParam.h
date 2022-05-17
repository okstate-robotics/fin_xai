// Author: Nathan Crosby
// Version: 0.0
// Date: 20210108
// ChangeLog:
//     0.0 -> initial creation

#ifndef HYPERPARAM_H
#define HYPERPARAM_H

namespace Hype {
  // This is going to be the class to define the hyperparameter use in the
  // FastMem project or others eventually
  class HyperParam {
    // private variables first
    private:
      double value;
      int min, max, divisor, step, direction, keyUpdate, turns, minTurns,
          set, numSteps;

    // Now the interface
    public:
      HyperParam();
      HyperParam(int mi, int ma, int div, int ns);
      void nextRand();
      void nextRand(double m);
      void setValue(double v);
      double getValue();
      void setDirection(int d);
      int getDirection();
      void setSet(int s);
      int getSet();
      int getMin();
      int getMax();
      int getDivisor();
      void reduceStep();
      void resetStep();
      int getStep();
      void setStep(int s);
      void setKeyUpdate(int k);
      int getKeyUpdate();
      void incrementTurns();
      void resetTurns();
      int getTurns();
      void setTurns(int t);
      int getMinTurns();
      void setMinTurns(int mt);
      void nextStep();
  }; // end HyperParam class
} // end namespace Hype

#endif
