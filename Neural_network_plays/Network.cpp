#include "Network.h"

Network::Network(const unsigned int inLayerSize, const unsigned int numHiddenlayer,
				 const unsigned hiddenLayerSize, const unsigned outLayerSize) : m_lvLayers(numHiddenlayer + 2)
{
	this->m_lvLayers.front().resize(inLayerSize);
	this->m_lvLayers.back().resize(outLayerSize);
	
	this->m_dvBias.resize(m_lvLayers.size() - 1);

	for (size_t i = 1; i < m_lvLayers.size()-1; i++)
	{
		this->m_lvLayers[i].resize(hiddenLayerSize);
	}
	for (size_t i = 1; i < m_lvLayers.size(); i++)
	{
		this->m_dvBias[i - 1].resize(this->m_lvLayers[i].size());
	}
	this->m_dvBias.back().resize(outLayerSize);

	this->m_wvWeightMatricies.resize(m_lvLayers.size() - 1);
	for (size_t i = 0; i < this->m_wvWeightMatricies.size(); i++)
	{
		this->m_wvWeightMatricies[i].resize(m_lvLayers[i + 1].size());
		for (auto& wmr : m_wvWeightMatricies[i]) {
			wmr.resize(m_lvLayers[i].size()); 
		}
	}
}

void mat_mul_vec(WeightMatrix& inMat, Layer& inLayer, Layer& outLayer) {
	//quick dim check
	if (inMat[0].size() != inLayer.size() || inMat.size() != outLayer.size()) throw std::exception("invalid size");

	for (size_t i = 0; i < inMat.size(); i++)
	{
		double val = 0.;
		for (size_t j = 0; j < inLayer.size(); j++)
		{
			val += inMat[i][j] * inLayer[j];
		}
		outLayer[i] = val;
	}
}

Layer operator+(Layer const& a, Layer const& b) {
	if (a.size() != b.size()) throw std::exception("unequal sized layers");

	Layer out(a.size());

	for (size_t i = 0; i < a.size(); i++)
	{
		out[i] = a[i] + b[i];
	}

	return out;
}

Layer Network::feed_through(Layer in)
{
	if (in.size() != this->m_lvLayers[0].size()) throw std::exception("layer too big or too smol");
	// input validierung bzw aktivirung
	// hier unter der annahme das der input valide ist

	this->m_lvLayers.front() = in;

	for (size_t i = 0; i < this->m_wvWeightMatricies.size(); i++)
	{
		mat_mul_vec(this->m_wvWeightMatricies[i], this->m_lvLayers[i], this->m_lvLayers[i + 1]);
		this->m_lvLayers[i+1] = this->m_lvLayers[i + 1] + this->m_dvBias[i];
		this->activate_layer(i + 1);
	}
	return this->m_lvLayers.back();
}

std::vector<WeightMatrix>& Network::get_weight_matricies()
{
	return this->m_wvWeightMatricies;
}

std::vector<Layer>& Network::get_layers()
{
	return this->m_lvLayers;
}

std::vector<std::vector<double>>& Network::get_biases()
{
	return this->m_dvBias;
}

// this merges the new coming weight adjustments into the existing by just adding them
void merge_delta_weights(std::vector<WeightMatrix>& target, std::vector<WeightMatrix> const& dw) {
	if (target.size() != dw.size())
		throw std::exception("size missmatch delta weights");

	for (size_t i = 0; i < target.size(); i++)
	{
		target[i] += dw[i];
	}
}

void merge_delta_bias(std::vector<Layer>& target, std::vector<Layer> const& db) {
	if (target.size() != db.size())
		throw std::exception("size missmatch delta bias");

	for (size_t i = 0; i < target.size(); i++)
	{
		target[i] = target[i] + db[i];
	}
}

double singleError;
void Network::train(std::vector<TrainingExample> const& trainingSet)
{
	double totalError = 0;
	for (auto& te : trainingSet) {
		this->stochastic_single_train(te);
		totalError += singleError;
	}	
		std::cout << totalError << '\n';
}

void Network::activate_layer(int i)
{
	for (double& d : this->m_lvLayers.at(i)) {
		d = this->m_afActivationFun(d);
	}
}

Layer calculate_layer_gradient(Layer l, ActivationFunctionPTR f) {
	Layer r;
	for (auto d : l) {
		r.push_back(f(d));
	}
	return r;
}

WeightMatrix wmTranspose(WeightMatrix const& wm) {
	WeightMatrix r;
	for (size_t i = 0; i < wm[0].size(); i++)
	{
		r.push_back(Layer(wm.size()));
	}

	for (size_t i = 0; i < wm.size(); i++)
	{
		for (size_t j = 0; j < wm[0].size(); j++)
		{
			r[j][i] = wm[i][j];
		}
	}

	return r;
}

void print(WeightMatrix const& wm) {
	for (size_t i = 0; i < wm.size(); i++)
	{
		std::cout << "| ";
		for (size_t j = 0; j < wm[0].size(); j++)
		{
			std::cout << wm[i][j] << " | ";
		}
		std::cout << '\n';
	}
	std::cout << std::endl;
}

void Network::stochastic_single_train(TrainingExample te)
{
	if (te.example.size() != this->m_lvLayers.front().size())
		throw std::exception("input size missmatch");
	if (te.expectedResult.size() != this->m_lvLayers.back().size())
		throw std::exception("output size missmatch");

	// directions in this case is the learningrate * error * direvative of the activation function
	auto matrix_nx1_X_matrix_1xm = [](Layer const& direction, Layer const& activations)->WeightMatrix {
		// create delta matrix
		WeightMatrix d(direction.size()); // rows
		for (auto& l : d)
			l.resize(activations.size()); //col width 

		for (size_t i = 0; i < direction.size(); i++)
		{
			for (size_t j = 0; j < activations.size(); j++)
			{
				d[i][j] = direction[i] * activations[j];
			}
		}

		return d;
	};

	Layer result = this->feed_through(te.example);

	std::vector<Layer> gradients;

	// calculate gradients
	for (size_t i = 1; i < this->m_lvLayers.size(); i++)
	{
		gradients.push_back(calculate_layer_gradient(this->m_lvLayers[i], this->m_afActivationFunDerivative));
	}

	Layer currentLayerError;

	//calculate output errors
	for (size_t i = 0; i < this->m_lvLayers.back().size(); i++)
	{
		currentLayerError.push_back(te.expectedResult[i] - result[i]);
	}

	singleError = 0;
	for (size_t i = 0; i < currentLayerError.size(); i++)
	{
		singleError += currentLayerError[i] * currentLayerError[i];
	}

	// calculate rest of the errors
	// start at size-2 because the last (ouput) layer has already been calculated
	// bigger 0 because we dont need to calculate an error for the input
	for (int i = this->m_lvLayers.size() - 2; i >= 0; i--)
	{
		/// since this sint so abvoious:
		/// lets look at a liniar case y = mx+b where the linar regression with the gradien descent brings us to
		/// dm = learningrate * error * (gradient, which is 1 in this case) * x	and
		/// db = learningrate * error * (gradient, which is 1 ...)
		/// this is actually quite easy to take over, 
		/// we can just view the m as our weights, x is the input from the previous layer and b being our bias
		/// which takes to 
		/// dweights = learningrate * error * privious activations		and
		/// dbias	 = learningrate * error								respectively
		/// but what is error?
		/// the error of a given layer is the factual numerical error * the gradient of the given layer
		/// 
		/// since the learningrate and error dont change, we can optimize a little bit and reuse the bias delta 
		/// to calculate the weight data
		
		/// calculate deltas and apply the changes
		///	  biasDelta				LR			*				(error)

		Layer biasDelta = this->m_dLearningRate * (gradients[i] * currentLayerError);
		WeightMatrix weightDelta = matrix_nx1_X_matrix_1xm(biasDelta, this->m_lvLayers[i]);
		//print(weightDelta);
		this->m_wvWeightMatricies[i] += weightDelta;
		this->m_dvBias[i] = this->m_dvBias[i] + biasDelta;

		//propagate back one layer
		Layer hiddenError(this->m_lvLayers[i].size());
		WeightMatrix weightsT = wmTranspose(this->m_wvWeightMatricies[i]);
		mat_mul_vec(weightsT, currentLayerError, hiddenError);
		currentLayerError = hiddenError;
	}
}
