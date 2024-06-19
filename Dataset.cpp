#include "Dataset.h"

Dataset::Dataset(std::vector<std::vector<double>>& dataset, Tip t) {

	podeli(dataset);
	skaliraj(t);
	csvier(_x_train, "xtrain");
	csvier(_y_train, "ytrain");
	csvier(_x_test,"xtest");
	csvier(_y_test,"ytest");
}

Dataset::~Dataset() {

	delete _x_train;
	delete _y_train;
	delete _x_test;
	delete _y_test;
}

int Dataset::random_num(int max) {
	return rand() % max;
}

void Dataset::podeli(std::vector<std::vector<double>>& dataset) {

	for (int i = 0; i < dataset.size(); i++) {
		int tmp = random_num(100);
		if (tmp < 30) {

			_x_test->push_back(dataset.at(i));

			_y_test->push_back({ _x_test->at(i).back() });

			_x_test->pop_back();
		}
		else {

			// umire ovde
			_x_train->push_back(dataset.at(i));

			_y_train->push_back({ _x_train->at(i).back() });

			_x_train->pop_back();
		}
	}
}

void Dataset::skaliraj(Tip t) {

	// adjusting x
	for (int i_kol = 0; i_kol < _x_train->at(0).size(); i_kol++) {

		double min_x = INFINITY;
		double max_x = -INFINITY;

		// trazenje min i max
		for (int i_red = 0; i_red < _x_train->size(); i_red++) {

			if (_x_train->at(i_red).at(i_kol) < min_x) min_x = _x_train->at(i_red).at(i_kol);
			if (_x_train->at(i_red).at(i_kol) > max_x) max_x = _x_train->at(i_red).at(i_kol);
		}

		// adjusting train
		for (int i_red = 0; i_red < _x_train->size(); i_red++) {

			_x_train->at(i_red).at(i_kol) = (_x_train->at(i_red).at(i_kol) - min_x) / (max_x - min_x);
		}

		for (int i_red = 0; i_red < _x_test->size(); i_red++) {

			_x_test->at(i_red).at(i_kol) = (_x_test->at(i_red).at(i_kol) - min_x) / (max_x - min_x);
		}
	}

	// adjusting y

	if (t == Tip::Regression) {

		for (int i_kol = 0; i_kol < _y_train->at(0).size(); i_kol++) {

			double min_y = INFINITY;
			double max_y = -INFINITY;

			// trazenje min i max
			for (int i_red = 0; i_red < _y_train->size(); i_red++) {

				if (_y_train->at(i_red).at(i_kol) < min_y) min_y = _y_train->at(i_red).at(i_kol);
				if (_y_train->at(i_red).at(i_kol) > max_y) max_y = _y_train->at(i_red).at(i_kol);
			}

			// adjusting train
			for (int i_red = 0; i_red < _y_train->size(); i_red++) {

				_y_train->at(i_red).at(i_kol) = (_y_train->at(i_red).at(i_kol) - min_y) / (max_y - min_y);
			}

			// adjusting test
			for (int i_red = 0; i_red < _y_test->size(); i_red++) {

				_y_test->at(i_red).at(i_kol) = (_y_test->at(i_red).at(i_kol) - min_y) / (max_y - min_y);
			}
		}
	}

	else if (t == Tip::Classification) {

		// y su sacuvani kao base10 brojevi, konvertujemo ih u custom vektore
		// kada imamo 4 opcije, 1 = 1 0 0 0 ; 2 = 0 1 0 0 ; 3 = 0 0 1 0 ; 4 = 0 0 0 1
		// trazimo max i na osnovu njega odredjujemo broj elemenata vektora

		// mozda olaksa nesto?
		int i_kol = 0;

		double max_y = -1;

		for (int i_red = 0; i_red < _y_train->size(); i_red++) {

			if (_y_train->at(i_red).at(i_kol) > max_y) max_y = _y_train->at(i_red).at(i_kol);
		}

		for (int i_red = 0; i_red < _y_test->size(); i_red++) {

			if (_y_test->at(i_red).at(i_kol) > max_y) max_y = _y_test->at(i_red).at(i_kol);
		}

		// adjusting train
		for (int i_red = 0; i_red < _y_train->size(); i_red++) {

			std::vector<double> vrednost;

			for (int i = 0; i < max_y; i++) {
				if (i + 1 == _y_train->at(i_red).at(i_kol))
					vrednost.push_back(1);
				else vrednost.push_back(0);
			}

			_y_train->at(i_red) = vrednost;
		}

		// adjusting test
		for (int i_red = 0; i_red < _y_test->size(); i_red++) {

			std::vector<double> vrednost1;

			for (int i = 0; i < max_y; i++) {
				if (i + 1 == _y_test->at(i_red).at(i_kol))
					vrednost1.push_back(1);
				else vrednost1.push_back(0);
			}

			_y_test->at(i_red) = vrednost1;
		}
	}
}

std::vector<std::vector<double>>* Dataset::get_x_train() {

	return _x_train;
}

std::vector<std::vector<double>>* Dataset::get_y_train() {

	return _y_train;
}

std::vector<std::vector<double>>* Dataset::get_x_test() {

	return _x_test;
}

std::vector<std::vector<double>>* Dataset::get_y_test() {

	return _y_test;
}

void Dataset::csvier(std::vector<std::vector<double>>* skup, std::string fileName) {

	std::cout << skup->size() << std::endl;

	std::ofstream File(fileName + ".csv");
	for (int i = 0; i < skup->size(); i++) {

		std::string line;
		for (int j = 0; j < skup->at(0).size(); j++) {

			line = line + std::to_string(skup->at(i).at(j)) + ",";
		}

		File << line << std::endl;
	}

	File.close();
}