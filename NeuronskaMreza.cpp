#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "NeuralNetwork.h"
#include "Dataset.h"
#include "Tip.h"

std::vector<std::vector<double>> readHousing(std::string fileName) {
    std::ifstream file(fileName);

    if (!file.is_open()) {

        std::cout << "Could not open the file " << fileName << std::endl;
        std::vector<std::vector<double>> meow;
        return meow;
    }

    std::vector<std::vector<double>> output;
    std::string str;

    std::string income;
    std::string age;
    std::string rooms;
    std::string bedrooms;
    std::string population;
    std::string price;

    bool i = true;

    while (getline(file, str)) {

        if (i) {

            getline(file, income, ',');
            getline(file, age, ',');
            getline(file, rooms, ',');
            getline(file, bedrooms, ',');
            getline(file, population, ',');
            getline(file, price, ',');

            output.push_back({ stod(income), stod(age), stod(rooms), stod(bedrooms), stod(population), stod(price) });
        }
        i = !i;
    }
    file.close();
    return output;
}

std::vector<std::vector<double>> readIris(std::string fileName) {
    std::ifstream file(fileName);

    if (!file.is_open()) {
        std::cout << "Could not open the file " << fileName << std::endl;
        std::vector<std::vector<double>> meow;
        return meow;
    }

    std::vector<std::vector<double>> output;

    std::string idrop;
    std::string slength;
    std::string swidth;
    std::string plength;
    std::string pwidth;
    std::string species;

    getline(file, idrop);
    while (getline(file, idrop, ',')) {
        getline(file, slength, ',');
        getline(file, swidth, ',');
        getline(file, plength, ',');
        getline(file, pwidth, ',');
        getline(file, species);

        double specie_num = 0;

        if (species == "Iris-setosa") specie_num = 1;
        else if (species == "Iris-versicolor") specie_num = 2;
        else if (species == "Iris-virginica") specie_num = 3;

        // std::cout << slength << " " << swidth << " " << plength << " " << pwidth << " " << specie_num << std::endl;
        output.push_back({ stod(slength), stod(swidth), stod(plength), stod(pwidth), specie_num });
    }

    file.close();
    return output;
}

int main()
{

    // std::vector<std::vector<double>> dataset = readHousing("housing.csv"); Tip t = Tip::Regression;

    std::vector<std::vector<double>> dataset = readIris("Iris.csv"); Tip t = Tip::Classification;


    Dataset d(dataset, t);

    NeuralNetwork* neuronska = new NeuralNetwork(t, { 5, 8, 5, 6, 5 }, d.get_x_train()->at(0).size(), d.get_y_train()->at(0).size(), 10e-6);

    neuronska->train(d.get_x_train(), d.get_y_train(), d.get_x_test(), d.get_y_test(), 200);

    delete(neuronska);

    std::cout << "Hello World!\n";
}