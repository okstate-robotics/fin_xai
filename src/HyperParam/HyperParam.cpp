// See ChangeLog in header file for details

// implementation of the HyperParam class

#include <stdlib.h>
#include <iostream>
#include "HyperParam.h"

Hype::HyperParam::HyperParam() {
  value = 0.0;
  min = 0;
  max = 10;
  divisor = 1;
  nextRand();
} // end default constructor

Hype::HyperParam::HyperParam(int mi, int ma, int div, int ns) {
  min = mi;
  max = ma;
  divisor = div;
  numSteps = ns;
  nextRand();
} // end constructor

// Main purpose of this class is to alter the hyperparameters each run
void Hype::HyperParam::nextRand() {
  if(max - min <= 0)
    value = min / (double) divisor;
  else
    value = (min + rand() % (max - min)) / (double) divisor;
  direction = 0;
  keyUpdate = -1;
  step = (max - min) / numSteps;
  turns = 0;
  minTurns = 0;
} // end nextRand()

// Special case to limit nextRand further
void Hype::HyperParam::nextRand(double m) {
  int parMax = (int) (m * divisor);

  // instead of using the built in max sub with the input
  if(parMax <= min)
    value = min;
  else 
    value = (min + rand() % (parMax - min)) / (double) divisor;

  direction = 0;
  keyUpdate = -1;
  step = (max - min) / numSteps;
  turns = 0;
  minTurns = 0;
} // end nextRand()

void Hype::HyperParam::setValue(double v) { value = v; }

double Hype::HyperParam::getValue() { return value; }

void Hype::HyperParam::setDirection(int d) { direction = d; }

int Hype::HyperParam::getDirection() { return direction; }

void Hype::HyperParam::setSet(int s) { set = s; }

int Hype::HyperParam::getSet() { return set; }

int Hype::HyperParam::getMin() { return min; }

int Hype::HyperParam::getMax() { return max; }

int Hype::HyperParam::getDivisor() { return divisor; }

void Hype::HyperParam::reduceStep() {
  step /= 2;
  if(step < 1)
    step = 1;
} // end reduceStep

void Hype::HyperParam::resetStep() {
  step = (max - min) / numSteps;
}

int Hype::HyperParam::getStep() { return step; }

void Hype::HyperParam::setStep(int s) { step = s; }

void Hype::HyperParam::setKeyUpdate(int k) { keyUpdate = k; }

int Hype::HyperParam::getKeyUpdate() { return keyUpdate; }

void Hype::HyperParam::incrementTurns() {
  turns++;
  reduceStep();
}

void Hype::HyperParam::resetTurns() { turns = 0; }

int Hype::HyperParam::getTurns() { return turns; }

void Hype::HyperParam::setTurns(int t) { turns = t; }

int Hype::HyperParam::getMinTurns() { return minTurns; }

void Hype::HyperParam::setMinTurns(int mt) { minTurns = mt; }

// next step is the new function that is going to more intelligently
// pick the next value for the particular hyper parameter. 
void Hype::HyperParam::nextStep() {
  double test = value + (direction * ((double)step / divisor));
  if( test > ((double)max/divisor))
    value = ((double)max/divisor);
  else if( test < ((double)min/divisor))
    value = ((double)min/divisor);
  else
    value = test;
} // end nextStep()
