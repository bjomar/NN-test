#pragma once

#include "common.h"

/// <summary>
/// a fully connected neural net
/// </summary>
class Network {
public:
	/// <summary>
	/// wrapps a single training example
	/// tuple::first beeing the input and ::second the expected outcome
	/// these *must* be the same size as the inputlayer and the outputlayer
	/// </summary>
	struct TrainingExample {
		Layer example, expectedResult;
	};

public:
	Network(const unsigned int inLayerSize, const unsigned int numHiddenlayer, 
			const unsigned hiddenLayerSize, const unsigned outLayerSize);

	Layer feed_through(Layer in);
	void train(std::vector<TrainingExample> const& trainingSet);

	std::vector<WeightMatrix>& get_weight_matricies();
	std::vector<Layer>& get_layers();
	std::vector<std::vector<double>>& get_biases();


	ActivationFunctionPTR m_afActivationFun = &sigmoid;
	// dsigmoid2 is used here because the network will save the layers in their activated state, 
	// removing the need to process them in the derivativew
	ActivationFunctionPTR m_afActivationFunDerivative = &dsigmoid2;
	ErrorFunctionPTR m_efErrorFunction = &identity;
	double m_dLearningRate = .1;

private:
	void activate_layer(int i);
	
	void stochastic_single_train(TrainingExample te);

	/// <summary>
	/// gewichte sind indiziert nach folgendem schema:
	/// 
	/// Wkj mit j = die aktivirung des jten nodes  und k = kte zielnode
	/// 
	/// W00 W01 .... W0j
	/// W10  .
	/// ...     .
	/// Wk0        . Wkj
	/// 
	/// </summary>
	std::vector<WeightMatrix> m_wvWeightMatricies;
	std::vector<Layer> m_lvLayers;
	std::vector<std::vector<double>> m_dvBias; // biases? bien? idk
};

