#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "NeuralNetwork.h"

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




int rand_num(int max) {
    return rand() % max;
}

std::vector<std::vector<std::vector<double>>> podeljen(std::vector<std::vector<double>> dataset) {

    std::vector<std::vector<std::vector<double>>> podeljeni(4);

    int xtrain = 0;
    int xtest = 2;
    int ytrain = 1;
    int ytest = 3;

    for (int i = 0; i < dataset.size(); i++) {
        if (rand_num(100) < 30) {
            podeljeni.at(xtest).push_back(dataset.at(i));
        }
        else {
            podeljeni.at(xtrain).push_back(dataset.at(i));
        }
    }

    // trening y
    for (int i = 0; i < podeljeni.at(xtrain).size(); i++) {
        double y = podeljeni.at(xtrain).at(i).back();
        podeljeni.at(ytrain).push_back({ y });
        podeljeni.at(xtrain).at(i).pop_back();
    }

    // test y
    for (int i = 0; i < podeljeni.at(xtest).size(); i++) {
        double y = podeljeni.at(xtest).at(i).back();
        podeljeni.at(ytest).push_back({ y });
        podeljeni.at(xtest).at(i).pop_back();
    }

    return podeljeni;
}

std::vector<std::vector<std::vector<double>>> skaliran(std::vector<std::vector<std::vector<double>>> podeljeni, bool regression) {

    // procicemo kroz jednu kolonu "matrice" doublova

    // ovo postoji da ne hardcodujem vrednosti jer mi je tako lakse da se zbunim
    int xtrain = 0;
    int xtest = 2;
    int ytrain = 1;
    int ytest = 3;


    for (int i_kol = 0; i_kol < podeljeni.at(xtrain).at(0).size(); i_kol++) {

        double min_x = INFINITY;
        double max_x = -INFINITY;

        // x_test i x_train ce imati isti broj doublova u koloni, 
        // sto znaci da mozemo da iskoristimo ovaj i_kol da bi skalirali i test, 
        // pa ne cuvamo maxove i minove sto smanjuje ulozeni trud

        // trazenje min i max
        for (int i_red = 0; i_red < podeljeni.at(xtrain).size(); i_red++) {

            if (podeljeni.at(xtrain).at(i_red).at(i_kol) < min_x) min_x = podeljeni.at(xtrain).at(i_red).at(i_kol);
            if (podeljeni.at(xtrain).at(i_red).at(i_kol) > max_x) max_x = podeljeni.at(xtrain).at(i_red).at(i_kol);
        }

        // adjusting train
        for (int i_red = 0; i_red < podeljeni.at(xtrain).size(); i_red++) {

            podeljeni.at(xtrain).at(i_red).at(i_kol) = (podeljeni.at(xtrain).at(i_red).at(i_kol) - min_x) / (max_x - min_x);
        }

        // adjusting test
        for (int i_red = 0; i_red < podeljeni.at(xtest).size(); i_red++) {

            podeljeni.at(xtest).at(i_red).at(i_kol) = (podeljeni.at(xtest).at(i_red).at(i_kol) - min_x) / (max_x - min_x);
        }
    }

    // adjustanje y
    // regresija ili klasifikacija
    if (regression) {

        for (int i_kol = 0; i_kol < podeljeni.at(ytrain).at(0).size(); i_kol++) {

            double min_y = INFINITY;
            double max_y = -INFINITY;

            // trazenje min i max
            for (int i_red = 0; i_red < podeljeni.at(ytrain).size(); i_red++) {

                if (podeljeni.at(ytrain).at(i_red).at(i_kol) < min_y) min_y = podeljeni.at(ytrain).at(i_red).at(i_kol);
                if (podeljeni.at(ytrain).at(i_red).at(i_kol) > max_y) max_y = podeljeni.at(ytrain).at(i_red).at(i_kol);
            }

            // adjusting train
            for (int i_red = 0; i_red < podeljeni.at(ytrain).size(); i_red++) {

                podeljeni.at(ytrain).at(i_red).at(i_kol) = (podeljeni.at(ytrain).at(i_red).at(i_kol) - min_y) / (max_y - min_y);
            }

            // adjusting test
            for (int i_red = 0; i_red < podeljeni.at(ytest).size(); i_red++) {

                podeljeni.at(ytest).at(i_red).at(i_kol) = (podeljeni.at(ytest).at(i_red).at(i_kol) - min_y) / (max_y - min_y);
            }
        }
    }
    else { // klasifikacija

        // trazimo max, sto ce biti broj neurona koji nam treba, tj dimenzionalnost vektora
        // posto postoji mogucnost da najveci parametar bude samo u testu/treningu, prolazimo kroz obe
        // nema negativnih vrednosti, ne trazimo min

        // za klasifikaciju odredjujemo kojoj grupi pripada, samo jedan rezultat
        // mozda moze da odredjuje za vise skupova grupa, ali ne bih

        int i_kol = 0;

        // nema negativnih vrednosti, ne zamajavamo se beskonacnostima
        double max_y = -1; // max vrednost klasifikacije

        for (int i_red = 0; i_red < podeljeni.at(ytrain).size(); i_red++) {

            if (podeljeni.at(ytrain).at(i_red).at(i_kol) > max_y) max_y = podeljeni.at(ytrain).at(i_red).at(i_kol);
        }

        for (int i_red = 0; i_red < podeljeni.at(ytest).size(); i_red++) {

            if (podeljeni.at(ytest).at(i_red).at(i_kol) > max_y) max_y = podeljeni.at(ytest).at(i_red).at(i_kol);
        }

        // adjusting train
        for (int i_red = 0; i_red < podeljeni.at(ytrain).size(); i_red++) {

            std::vector<double> vrednost(max_y);

            for (int i = 0; i < max_y; i++) {
                if (i + 1 == podeljeni.at(ytrain).at(i_red).at(i_kol))
                    vrednost.at(i) = 1;
                else vrednost.at(i) = 0;
            }

            podeljeni.at(ytrain).at(i_red) = vrednost;
        }

        // adjusting test
        for (int i_red = 0; i_red < podeljeni.at(ytest).size(); i_red++) {

            std::vector<double> vrednost1(max_y);

            for (int i = 0; i < max_y; i++) {
                if (i + 1 == podeljeni.at(ytest).at(i_red).at(i_kol))
                    vrednost1.at(i) = 1;
                else vrednost1.at(i) = 0;
            }

            podeljeni.at(ytest).at(i_red) = vrednost1;
        }

    }

    return podeljeni;
}

std::vector<std::vector<std::vector<double>>> podesiPodatke(bool regresija) {

    std::vector<std::vector<double>> dataset;

    if (regresija) dataset = readHousing("housing.csv");
    else dataset = readIris("Iris.csv");

    std::vector<std::vector<std::vector<double>>> podela = podeljen(dataset);
    return skaliran(podela, regresija);
}

// ovo radi samo za regresiju
void dataToFileRegression(bool train, std::vector<std::vector<std::vector<double>>> data) {

    int x = -1; int y = -1;
    std::string fileName;

    if (train) { x = 0; y = 1; fileName = "trainout.csv"; }
    else { x = 2; y = 3; fileName = "testout.csv"; }

    std::ofstream TrainFile(fileName);

    for (int i = 0; i < data.at(x).size(); i++) {
        std::string line;
        for (int j = 0; j < data.at(x).at(0).size(); j++) {

            line = line + std::to_string(data.at(x).at(i).at(j)) + ",";

        }
        line = line + std::to_string(data.at(y).at(i).at(0));

        TrainFile << line << std::endl;

    }

    TrainFile.close();
}

int main()
{

    std::vector<std::vector<std::vector<double>>> traintest = podesiPodatke(true);

    std::vector<std::vector<double>> x_train = traintest.at(0);
    std::vector<std::vector<double>> y_train = traintest.at(1);
    std::vector<std::vector<double>> x_test = traintest.at(2);
    std::vector<std::vector<double>> y_test = traintest.at(3);

    NeuralNetwork* neuronska = new NeuralNetwork({ 5, 5, 5, 5, 5 }, x_train.at(0).size(), y_train.at(0).size(), 10e-6);

    neuronska->train(x_train, y_train, x_test, y_test, 200);

    std::cout << "Hello World!\n";
}