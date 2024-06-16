#pragma once

#include "Layer.h"
#include <iostream>

class NeuralNetwork
{
private:
	std::vector<Layer*>* _layers;
	double _learning_rate;

public:
	NeuralNetwork(const std::vector<int>& hiddenLayerSpec, int numFeatures, int numClasses, double lr);

	~NeuralNetwork();
	NeuralNetwork(const NeuralNetwork& ann);
	NeuralNetwork& operator =(const NeuralNetwork& ann);

	// enkapsulirati podatke
	std::vector<double> forward_propagate(std::vector<double>& input);

	// backpropagaion
	// input -> feature vector
	// output -> results (class labels...)
	void back_propagate(std::vector<double>& input, std::vector<double>& output);
	void update_weights(std::vector<double>& input, std::vector<double>& output);

	void train(std::vector<std::vector<double>>& input, std::vector<std::vector<double>>& output,
		std::vector<std::vector<double>>& test_input,
		std::vector<std::vector<double>>& test_output, int epochs);
	double test(std::vector<std::vector<double>>& input, std::vector<std::vector<double>>& output);
};



