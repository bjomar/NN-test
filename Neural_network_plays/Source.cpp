#include <iostream>
#include <algorithm>
#include "Network.h"

#define frand (float)rand()/(float)(RAND_MAX/2)

void XOR () {
	Network n = Network(2, 1, 2, 1);
	auto& biases = n.get_biases();
	auto& weights = n.get_weight_matricies();

	n.m_afActivationFun = [](double x) ->double {return fmax(0, x); };

	std::cout << "weightmatricies dimensions:\n";
	for (size_t i = 0; i < weights.size(); i++)
	{
		std::cout << '\t' << weights[i].size() << " x " << weights[i][0].size() << '\n';
	}
	std::cout << "biases dimensions:\n";
	for (size_t i = 0; i < weights.size(); i++)
	{
		std::cout << '\t' << biases[i].size() << '\n';
	}

	// first index is the nth weightmatrix, 2nd is the nth matrixes yth row and 3rd is the zth column
	weights[0][0][0] = 1;
	weights[0][0][1] = 1;
	weights[0][1][0] = 1;
	weights[0][1][1] = 1;

	weights[1][0][0] = 1;
	weights[1][0][1] = -2;

	biases[0][0] = 0;
	biases[0][1] = -1;
	biases[1][0] = 0;

	try
	{
		auto res = n.feed_through({ 0,0 });
		std::cout << "result: " << res[0] << '\n';
		res = n.feed_through({ 0,1 });
		std::cout << "result: " << res[0] << '\n';
		res = n.feed_through({ 1,0 });
		std::cout << "result: " << res[0] << '\n';
		res = n.feed_through({ 1,1 });
		std::cout << "result: " << res[0] << '\n';
	}
	catch (const std::exception& e)
	{
		std::cout << e.what();
	}
}

void testTrain_XOR() {
	Network::TrainingExample te1;
	te1.example = { 0,0 };
	te1.expectedResult = { 0 };

	Network::TrainingExample te2;
	te2.example = { 0,1 };
	te2.expectedResult = { 1 };

	Network::TrainingExample te3;
	te3.example = { 1,0 };
	te3.expectedResult = { 1 };

	Network::TrainingExample te4;
	te4.example = { 1,1 };
	te4.expectedResult = { 0 };

	std::vector<Network::TrainingExample> trainingSet = { te1, te2, te3, te4 };

	Network n = Network(2, 1, 5, 1);
	n.m_dLearningRate = .1;
	auto& biases = n.get_biases();
	auto& weights = n.get_weight_matricies();

	std::cout << "weightmatricies dimensions:\n";
	for (size_t i = 0; i < weights.size(); i++)
	{
		std::cout << '\t' << weights[i].size() << " x " << weights[i][0].size() << '\n';
	}
	std::cout << "biases dimensions:\n";
	for (size_t i = 0; i < weights.size(); i++)
	{
		std::cout << '\t' << biases[i].size() << '\n';
	}

	// first index is the nth weightmatrix, 2nd is the nth matrixes yth row and 3rd is the zth column
	// initialize random
	weights[0][0][0] = frand;
	weights[0][0][1] = frand;
	weights[0][1][0] = -frand;
	weights[0][1][1] = frand;
	
	weights[1][0][0] = -frand;
	weights[1][0][1] = frand;

	biases[0][0] = -frand;
	biases[0][1] = frand;
	biases[1][0] = -frand;

	auto res = n.feed_through({ 0,0 });
	std::cout << "result: " << res[0] << '\n';
	res = n.feed_through({ 0,1 });
	std::cout << "result: " << res[0] << '\n';
	res = n.feed_through({ 1,0 });
	std::cout << "result: " << res[0] << '\n';
	res = n.feed_through({ 1,1 });
	std::cout << "result: " << res[0] << '\n';

	for (size_t i = 0; i < 100000; i++)
	{
		n.train({trainingSet[rand()%4]});
	}

	res = n.feed_through({ 0,0 });
	std::cout << "0,0 result: " << res[0] << '\n';
	res = n.feed_through({ 0,1 });
	std::cout << "0,1 result: " << res[0] << '\n';
	res = n.feed_through({ 1,0 });
	std::cout << "1,0 result: " << res[0] << '\n';
	res = n.feed_through({ 1,1 });
	std::cout << "1,1 result: " << res[0] << '\n';
}
#include <ctime>
void testTrain_XAND() {
	srand(time(NULL));

	Network::TrainingExample te1;
	te1.example = { 0,0 };
	te1.expectedResult = { 1,0 };

	Network::TrainingExample te2;
	te2.example = { 0,1 };
	te2.expectedResult = { 0,1 };

	Network::TrainingExample te3;
	te3.example = { 1,0 };
	te3.expectedResult = { 0,1 };

	Network::TrainingExample te4;
	te4.example = { 1,1 };
	te4.expectedResult = { 1,0 };

	std::vector<Network::TrainingExample> trainingSet = { te1, te2, te3, te4 };

	Network n = Network(2, 2, 6, 2);
	n.m_dLearningRate = .1;
	n.m_afActivationFun = &tanH;
	n.m_afActivationFunDerivative = &dtanh;
	n.m_efErrorFunction = &MSE;
	auto& biases = n.get_biases();
	auto& weights = n.get_weight_matricies();

	std::cout << weights.size() << " weightmatricies dimensions:\n";
	for (size_t i = 0; i < weights.size(); i++)
	{
		std::cout << '\t' << weights[i].size() << " x " << weights[i][0].size() << '\n';
	}
	std::cout << biases.size() << " biases dimensions:\n";
	for (size_t i = 0; i < weights.size(); i++)
	{
		std::cout << '\t' << biases[i].size() << '\n';
	}

	// first index is the nth weightmatrix, 2nd is the nth matrixes yth row and 3rd is the zth column
	// initialize random
	for (size_t i = 0; i < weights.size(); i++)
	{
		for (size_t j = 0; j < weights[i].size(); j++) // rows
		{
			for (size_t k = 0; k < weights[i][j].size(); k++)
			{
				weights[i][j][k] = rand() % 2 ? frand : -frand;
			}
		}
	}

	for (size_t i = 0; i < biases.size(); i++)
	{
		for (size_t j = 0; j < biases[i].size(); j++)
		{
			biases[i][j] = rand() % 2 ? frand : -frand;
		}
	}

	auto res = n.feed_through({ 0,0 });
	std::cout << "0,0 result: " << res[0] << res[1] << '\n';
	res = n.feed_through({ 0,1 });
	std::cout << "0,1 result: " << res[0] << res[1] << '\n';
	res = n.feed_through({ 1,0 });
	std::cout << "1,0 result: " << res[0] << res[1] << '\n';
	res = n.feed_through({ 1,1 });
	std::cout << "1,1 result: " << res[0] << res[1] << '\n';

	for (size_t i = 0; i < 3000; i++)
	{
		std::random_shuffle(trainingSet.begin(), trainingSet.end());
		n.train(trainingSet);
	}

	res = n.feed_through({ 0,0 });
	std::cout << "0,0 result: " << res[0] << '|' << res[1] << '\n';
	res = n.feed_through({ 0,1 });
	std::cout << "0,1 result: " << res[0] << '|' << res[1] << '\n';
	res = n.feed_through({ 1,0 });
	std::cout << "1,0 result: " << res[0] << '|' << res[1] << '\n';
	res = n.feed_through({ 1,1 });
	std::cout << "1,1 result: " << res[0] << '|' << res[1] << '\n';
}

void testTrain_multiplelogic() {
	srand(time(NULL));

	Network::TrainingExample te1;
	te1.example = { 0,0 };
	te1.expectedResult = { 0,1,0,0 };

	Network::TrainingExample te2;
	te2.example = { 0,1 };
	te2.expectedResult = { 1,0,1,0 };

	Network::TrainingExample te3;
	te3.example = { 1,0 };
	te3.expectedResult = { 1,0,1,0 };

	Network::TrainingExample te4;
	te4.example = { 1,1 };
	te4.expectedResult = { 0,1,1,1 };

	std::vector<Network::TrainingExample> trainingSet = { te1, te2, te3, te4 };

	Network n = Network(2, 2, 8, 4);
	n.m_dLearningRate = .1;
	n.m_afActivationFun = &tanH;
	n.m_afActivationFunDerivative = &dtanh;
	n.m_efErrorFunction = &MSE;
	auto& biases = n.get_biases();
	auto& weights = n.get_weight_matricies();

	std::cout << weights.size() << " weightmatricies dimensions:\n";
	for (size_t i = 0; i < weights.size(); i++)
	{
		std::cout << '\t' << weights[i].size() << " x " << weights[i][0].size() << '\n';
	}
	std::cout << biases.size() << " biases dimensions:\n";
	for (size_t i = 0; i < weights.size(); i++)
	{
		std::cout << '\t' << biases[i].size() << '\n';
	}

	// first index is the nth weightmatrix, 2nd is the nth matrixes yth row and 3rd is the zth column
	// initialize random
	for (size_t i = 0; i < weights.size(); i++)
	{
		for (size_t j = 0; j < weights[i].size(); j++) // rows
		{
			for (size_t k = 0; k < weights[i][j].size(); k++)
			{
				weights[i][j][k] = rand() % 2 ? frand : -frand;
			}
		}
	}

	for (size_t i = 0; i < biases.size(); i++)
	{
		for (size_t j = 0; j < biases[i].size(); j++)
		{
			biases[i][j] = rand() % 2 ? frand : -frand;
		}
	}

	auto res = n.feed_through({ 0,0 });
	std::cout << "0,0 result: " << res[0] << '|' << res[1] << '|' << res[2] << '|' << res[3] << '\n';
	res = n.feed_through({ 0,1 });
	std::cout << "0,1 result: " << res[0] << '|' << res[1] << '|' << res[2] << '|' << res[3] << '\n';
	res = n.feed_through({ 1,0 });
	std::cout << "1,0 result: " << res[0] << '|' << res[1] << '|' << res[2] << '|' << res[3] << '\n';
	res = n.feed_through({ 1,1 });
	std::cout << "1,1 result: " << res[0] << '|' << res[1] << '|' << res[2] << '|' << res[3] << '\n';


	for (size_t i = 0; i < 3000; i++)
	{
		std::random_shuffle(trainingSet.begin(), trainingSet.end());
		n.train(trainingSet);
	}

	res = n.feed_through({ 0,0 });
	std::cout << "0,0 result: " << res[0] << '|' << res[1] << '|' << res[2] << '|' << res[3] << '\n';
	res = n.feed_through({ 0,1 });
	std::cout << "0,1 result: " << res[0] << '|' << res[1] << '|' << res[2] << '|' << res[3] << '\n';
	res = n.feed_through({ 1,0 });
	std::cout << "1,0 result: " << res[0] << '|' << res[1] << '|' << res[2] << '|' << res[3] << '\n';
	res = n.feed_through({ 1,1 });
	std::cout << "1,1 result: " << res[0] << '|' << res[1] << '|' << res[2] << '|' << res[3] << '\n';
}

int main(int argc, char** argv) {
	testTrain_multiplelogic();
}