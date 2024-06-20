#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(Tip t, const std::vector<int>& hiddenLayerSpec, int numFeatures, int numClasses, double lr) {

	_learning_rate = lr;
	_layers = new std::vector<Layer*>();

	for (int i = 0; i < hiddenLayerSpec.size(); i++) {

		// input_layer
		if (i == 0) {
			_layers->push_back(new Layer(t, numFeatures, hiddenLayerSpec[i]));
		}
		// hidden layers
		else {
			_layers->push_back(new Layer(t, hiddenLayerSpec[i - 1], hiddenLayerSpec[i]));
		}
	}

	// output layer
	_layers->push_back(new Layer(t, hiddenLayerSpec.back(), numClasses));
}

NeuralNetwork::~NeuralNetwork() {

	for (int i = 0; i < _layers->size(); i++)
		delete _layers->at(i);

	delete _layers;
}
NeuralNetwork::NeuralNetwork(const NeuralNetwork& ann) {
	_learning_rate = ann._learning_rate;
	_layers = new std::vector<Layer*>();

	for (int i = 0; i < ann._layers->size(); i++)
		_layers->push_back(ann._layers->at(i)->Clone());
}
NeuralNetwork& NeuralNetwork::operator =(const NeuralNetwork& ann) {

	if (this == &ann)
		return *this;

	_learning_rate = ann._learning_rate;
	for (int i = 0; i < _layers->size(); i++)
		delete _layers->at(i);

	_layers->clear();
	for (int i = 0; i < ann._layers->size(); i++)
		_layers->push_back(ann._layers->at(i)->Clone());

	return *this;
}

// enkapsulirati podatke
std::vector<double> NeuralNetwork::forward_propagate(std::vector<double>& input) {

	std::vector<double>* inputVec = &input;
	for (int i = 0; i < _layers->size(); i++) {
		_layers->at(i)->forward_propagate(inputVec);
		inputVec = _layers->at(i)->get_outputs();
	}

	// nije pametno
	return *(_layers->back()->get_outputs());
}

void NeuralNetwork::back_propagate(std::vector<double>& input, std::vector<double>& output) {

	// racunamo delte
	for (int i = _layers->size() - 1; i >= 0; i--) {

		// 1. korak je racunanje gresaka -> koliko rezulat odstupa od ocekivanog
		std::vector<double> errs;
		// unutrasnji slojevi
		if (i != _layers->size() - 1) {
			// izracunavanje uticaja neurona j na sve neurone k u sledecem sloju
			for (int j = 0; j < _layers->at(i)->get_layer_size(); j++) {
				double err = 0;

				// prolazimo kroz sve neurone u sledecem sloju
				for (int k = 0; k < _layers->at(i + 1)->get_layer_size(); k++) {
					err +=
						_layers->at(i + 1)->get_neurons()->at(k)->get_weights()->at(j)
						*
						_layers->at(i + 1)->get_neurons()->at(k)->get_delta();
				}

				// pamtimo kumulativni uticaj neuron j na sve neurone k
				errs.push_back(err);
			}
		}
		// izlazni sloj
		else {
			// ceo sloj
			for (int j = 0; j < _layers->at(i)->get_layer_size(); j++) {
				double err = (_layers->at(i)->get_outputs()->at(j) - output[j]);
				errs.push_back(err);
			}
		}

		// 2. korak -> racunanje samih delti, tj. gradijenta
		// pamtimo delte u svakom neuronu
		for (int j = 0; j < _layers->at(i)->get_layer_size(); j++) {
			Neuron* n = _layers->at(i)->get_neurons()->at(j);
			// proverite da li je u strukturi get_output izlaz neurona
			double delta = errs.at(j) * n->transfer_derivative(n->get_output());
			n->set_delta(delta);
		}
	}
}

//void NeuralNetwork::update_weights(std::vector<double>& input, std::vector<double>& output) {
//
//	std::vector<double> inputs = input;
//	for (int i = 0; i < _layers->size(); i++) {
//
//		// razdvajanje "ulaznog" sloja od ostalih
//		// podaci za ulazni sloj su dati argumenom input
//		// dok za unutrasnje slojeve mi moramo da proglasimo sta su ulazi
//		if (i != 0) {
//			for (int j = 0; j < _layers->at(i)->get_neurons()->size(); j++) {
//				inputs.push_back(_layers->at(i)->get_neurons()->at(j)->get_output());
//			}
//		}
//
//		// gradijentni spust
//		for (int j = 0; j < _layers->at(i)->get_neurons()->size(); j++) {
//			Neuron* n = _layers->at(i)->get_neurons()->at(j);
//
//			// popravka svih tezina gradijentnim spustom
//			for (int k = 0; k < inputs.size(); k++) {
//				n->get_weights()->at(k) += this->_learning_rate * n->get_delta() * inputs.at(k);
//				// std::cout << n->get_delta() << std::endl;
//			}
//
//			// bias
//			n->get_weights()->back() += this->_learning_rate * n->get_delta();
//			
//		}
//
//		inputs.clear();
//	}
//}

void NeuralNetwork::update_weights(std::vector<double>& input, std::vector<double>& output) {

	// Initial input to the network
	std::vector<double> inputs = input;
	std::vector<double> new_inputs;

	// Loop through each layer
	for (int i = 0; i < _layers->size(); i++) {

		// Create a new vector for storing inputs for the next layer
		new_inputs.clear();

		// Loop through each neuron in the current layer
		for (int j = 0; j < _layers->at(i)->get_neurons()->size(); j++) {
			Neuron* n = _layers->at(i)->get_neurons()->at(j);

			// Update weights using gradient descent
			for (int k = 0; k < inputs.size(); k++) {
				n->get_weights()->at(k) -= this->_learning_rate * n->get_delta() * inputs.at(k);
			}

			// Update the bias
			n->get_weights()->back() -= this->_learning_rate * n->get_delta();

			// Collect the output of the current neuron to be used as input for the next layer
			new_inputs.push_back(n->get_output());
		}

		// Set inputs to the outputs of the current layer for the next iteration
		inputs = new_inputs;
	}
}

void NeuralNetwork::train(std::vector<std::vector<double>>* input,
	std::vector<std::vector<double>>* output,
	std::vector<std::vector<double>>* test_input,
	std::vector<std::vector<double>>* test_output, int epochs) {

	std::ofstream trainFile("train_err.txt");
	std::ofstream testFile("test_err.txt");

	for (int i = 0; i < epochs; i++) {
		double totalErr = 0;
		//int j = 0;
		//for(int z = 0; z < input->size();z++) {
		for (int j = 0; j < input->size(); j++) {
			double err = 0;
			std::vector<double> pred = this->forward_propagate(input->at(j));
			for (int k = 0; k < output->at(j).size(); k++) {
				err += (pred[k] - output->at(j)[k]* (pred[k] - output->at(j)[k]));
			}
			totalErr += err;

			this->back_propagate(input->at(j), output->at(j));
			this->update_weights(input->at(j), output->at(j));
		}
		totalErr /= input->size();

		// test set, train set
		double testError = test(test_input, test_output);
		std::cout << "Training epoch: " << (i + 1) << " MSE: " << totalErr
			<< "Test MSE: " << testError << std::endl;

		trainFile << totalErr << std::endl;
		testFile << testError << std::endl;
		
	}

	trainFile.close();
	testFile.close();
}

double NeuralNetwork::test(std::vector<std::vector<double>>* input, std::vector<std::vector<double>>* output) {

	double totalErr = 0;
	//int j = 0;
	//for(int k = 0; k < input->size(); k++) {
	for (int j = 0; j < input->size(); j++) {
		double err = 0;
		std::vector<double> pred = this->forward_propagate(input->at(j));
		for (int k = 0; k < output->at(j).size(); k++) {
			err += (pred[k] - output->at(j)[k]) * (pred[k] - output->at(j)[k]);
		}
		totalErr += err;
	}
	totalErr /= input->size();

	return totalErr;
}