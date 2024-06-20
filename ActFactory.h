#pragma once
#include <math.h>
#include "Tip.h"

class ActFun
{
public:
	virtual double activation(double x) = 0;
	virtual double derivative(double act) = 0;
};

class ReLu : public ActFun {
public:
	double activation(double x);
	double derivative(double act);
};

class Sigmoid : public ActFun {
public:
	double activation(double x);
	double derivative(double act);
};

class LeakyReLu : public ActFun {
private:
	double _alpha = 0.01;
public:
	double activation(double x);
	double derivative(double act);
};

class ActFactory {
public:
	ActFun* createAct(Tip t);
};

