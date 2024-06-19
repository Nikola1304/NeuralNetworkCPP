#pragma once
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>

enum class Tip
{
	Regression,
	Classification
};

class Dataset
{
private:
	std::vector<std::vector<double>>* _x_train;
	std::vector<std::vector<double>>* _y_train;
	std::vector<std::vector<double>>* _x_test;
	std::vector<std::vector<double>>* _y_test;

	int random_num(int max);

	void podeli(std::vector<std::vector<double>>& dataset);
	void skaliraj(Tip t);

public:
	Dataset(std::vector<std::vector<double>>& dataset, Tip t);
	~Dataset();

	// ne treba konstruktor kopije

	std::vector<std::vector<double>>* get_x_train();
	std::vector<std::vector<double>>* get_y_train();
	std::vector<std::vector<double>>* get_x_test();
	std::vector<std::vector<double>>* get_y_test();

	void csvier(std::vector<std::vector<double>>* skup, std::string fileName);
};