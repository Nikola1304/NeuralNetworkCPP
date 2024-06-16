#include "Neuron.h"
#include <cstdlib>
#include <cmath>   

Neuron::Neuron(int prevLayerSize) {
    _weights = new std::vector<double>(prevLayerSize + 1, 0.0); 
    _output = 0;
    _delta = 0;
    _activation = 0;
}

Neuron::~Neuron() {
    delete _weights;
}

Neuron::Neuron(const Neuron& nn) {
    this->_delta = nn._delta;
    this->_output = nn._output;
    _activation = nn._activation;

    _weights = new std::vector<double>(*nn._weights); 
}

Neuron& Neuron::operator=(const Neuron& nn) {
    if (this == &nn)
        return *this;

    this->_delta = nn._delta;
    this->_output = nn._output;
    _activation = nn._activation;

    *_weights = *nn._weights; 

    return *this;
}

double Neuron::rand_num(double min, double max) {
    double x = static_cast<double>(rand()) / RAND_MAX;
    return x * (max - min) + min;
}

void Neuron::initialize_weights(int prevLayerSize) {
    _weights->resize(prevLayerSize + 1); 
    for (int i = 0; i < prevLayerSize + 1; i++) {
        _weights->at(i) = rand_num(-1, 1); 
    }
}

double Neuron::get_output() {
    return _output;
}

void Neuron::set_output(double x) {
    _output = x;
}

double Neuron::get_delta() {
    return _delta;
}

void Neuron::set_delta(double x) {
    _delta = x;
}

std::vector<double>* Neuron::get_weights() {
    return _weights;
}

Neuron* Neuron::Clone() {
    return new Neuron(*this);
}

double Neuron::get_activation() {
    return _activation;
}

void Neuron::set_activation(double x) {
    _activation = x;
}

double Neuron::inner_prod(std::vector<double>* input) {
    double rez = _weights->back(); // Bias
    for (size_t i = 0; i < input->size(); i++) {
        rez += _weights->at(i) * input->at(i);
    }
    _output = rez;
    return rez;
}

double Neuron::transfer(double x) {
    _activation = 1.0 / (1.0 + exp(-x)); // Sigmoid
    return _activation; // Sigmoid
}

double Neuron::transfer_derivative(double act) {
    // act = sigma(x);
    // sigma'(x) = sigma(x) * (1 - sigma(x))
    return act * (1 - act);

    // act = relu(x)
    // return act > 0 ? 1 : 0; // ReLU
}
