#include "Layer.h"
#include <iostream>

Layer::Layer(int prevLayerSize, int layerSize) {

	_neurons = new std::vector<Neuron*>();
	_outputs = new std::vector<double>();
	_layerSize = layerSize;

	for (int i = 0; i < _layerSize; i++) {
		_neurons->push_back(new Neuron(prevLayerSize));
		_outputs->push_back(0.0);
	}
}

Layer::~Layer() {

	for (int i = 0; i < _layerSize; i++)
		delete _neurons->at(i);

	delete _outputs;
	delete _neurons;
}
Layer::Layer(const Layer& ll) {

	_neurons = new std::vector<Neuron*>();
	_outputs = new std::vector<double>();
	_layerSize = ll._layerSize;

	for (int i = 0; i < _layerSize; i++) {
		_neurons->push_back(ll._neurons->at(i)->Clone());
		_outputs->push_back(ll._outputs->at(i));
	}
}
Layer& Layer::operator =(const Layer& ll) {

	if (this == &ll)
		return *this;

	for (int i = 0; i < _layerSize; i++)
		delete _neurons->at(i);

	_outputs->clear();
	_neurons->clear();

	_layerSize = ll._layerSize;
	for (int i = 0; i < _layerSize; i++) {
		_neurons->push_back(ll._neurons->at(i)->Clone());
		_outputs->push_back(ll._outputs->at(i));
	}

	return *this;
}

std::vector<Neuron*>* Layer::get_neurons() {
	return _neurons;
}
std::vector<double>* Layer::get_outputs() {
	return _outputs;
}
int Layer::get_layer_size() {
	return _layerSize;
}

Layer* Layer::Clone() {

	return new Layer(*this);
}

void Layer::forward_propagate(std::vector<double>* input) {

	for (int i = 0; i < _neurons->size(); i++) {
		double lin_komb = _neurons->at(i)->inner_prod(input);
		_neurons->at(i)->set_output(lin_komb);

		double act = _neurons->at(i)->transfer(lin_komb);
		_neurons->at(i)->set_activation(act);

		(*_outputs)[i] = act;
	}
}