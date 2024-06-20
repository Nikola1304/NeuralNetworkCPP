#include "ActFactory.h"


double ReLu::activation(double x) {

	if (x > 0)
		return x;
	else
		return 0;
}

double ReLu::derivative(double act) {

	return act > 0 ? 1 : 0;
}

double Sigmoid::activation(double x) {

	return 1.0 / (1.0 + exp(-x));
}

double Sigmoid::derivative(double act) {

	return act * (1 - act);
}

double LeakyReLu::activation(double x) {
	if (x > 0) return x;
	else return _alpha * x;
}

double LeakyReLu::derivative(double act) {

	// return act > 0 ? 1 : _alpha;
	if (act > 0) return 1;
	else if (act < 0) return _alpha;
	else return 0;
}

ActFun* ActFactory::createAct(Tip t) {
	if (t == Tip::Regression)
		return new LeakyReLu();
	else if (t == Tip::Classification)
		return new Sigmoid();
}