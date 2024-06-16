#pragma once

#include "Neuron.h"

class Layer
{
private:
	std::vector<Neuron*>* _neurons;
	std::vector<double>* _outputs;
	int _layerSize; // bez biasa

public:
	Layer(int prevLayerSize, int layerSize);

	~Layer();
	Layer(const Layer& ll);
	Layer& operator =(const Layer& ll);

	std::vector<Neuron*>* get_neurons();
	std::vector<double>* get_outputs();
	int get_layer_size();

	Layer* Clone();

	void forward_propagate(std::vector<double>* input);
};


