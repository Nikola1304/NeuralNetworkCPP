#pragma once

#include <vector>
#include <cstdlib>
#include <cmath>

class Neuron
{
private:
	std::vector<double>* _weights;
	double _output;
	double _activation;
	double _delta;

	void initialize_weights(int prevLayerSize);
	double rand_num(double min, double max);

public:
	Neuron(int prevLayerSize);

	~Neuron();
	Neuron(const Neuron& nn);
	Neuron& operator =(const Neuron& nn);

	double get_output();
	void set_output(double x);

	double get_delta();
	void set_delta(double x);

	double get_activation();
	void set_activation(double x);

	// linearna kombinacija
	double inner_prod(std::vector<double>* input);
	// primena aktivacione fukcije na output
	double transfer(double x);
	// vrednost izvoda aktivacione funkcije u tacki x
	double transfer_derivative(double x);

	std::vector<double>* get_weights();

	Neuron* Clone();
};

