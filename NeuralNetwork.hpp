#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

class Neuron {
private:
  vector<double> weights;
  double bias;
  vector<double> lastInput;
  double lastOutput;
  double delta;

  double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

  double sigmoidDerivative(double x) { return x * (1.0 - x); }

public:
  Neuron(int inputSize) {
    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<double> dist(-1.0, 1.0);

    weights.resize(inputSize);

    for (int i = 0; i < inputSize; i++) {
      weights[i] = dist(gen);
    }

    bias = dist(gen);
  }

  double forward(const vector<double> &input) {
    lastInput = input;
    double sum = bias;

    for (int i = 0; i < input.size(); i++) {
      sum += weights[i] * input[i];
    }

    lastOutput = sigmoid(sum);
    return lastOutput;
  }
  void computeHiddenDelta(double downstream) {
    delta = sigmoidDerivative(lastOutput) * downstream;
  }

  void train(const vector<double> &input, const double target,
             const double lr) {
    double prediction = forward(input);
    double err = target - prediction;
    double gradient = err * sigmoidDerivative(prediction);

    for (int i = 0; i < weights.size(); i++) {
      weights[i] += lr * gradient * input[i];
    }
    bias += lr * gradient;
  }

  void computeOutputDelta(double target) {
    double error = target - lastOutput;
    delta = error * sigmoidDerivative(lastOutput);
  }

  void updateWeights(double lr) {
    for (int i = 0; i < weights.size(); i++) {
      weights[i] += lr * delta * lastInput[i];
    }

    bias += lr * delta;
  }

  void setWeight(int idx, double value) { weights[idx] = value; }

  void setBias(double value) { bias = value; }
  double getOutput() const { return lastOutput; }
  double getWeight(int idx) { return weights[idx]; }
  double getDelta() const { return delta; }

  const vector<double> &getWeights() const { return weights; }
  double getBias() const { return bias; }

  void loadParameters(const vector<double> &newWeights, const double newBias) {
    weights = newWeights;
    bias = newBias;
  }
};

class Layer {
private:
  vector<Neuron> neurons;
  void computeHiddenDeltas(Layer &nextLayer, int hiddenIndex) {
    double downstream = 0;

    for (int i = 0; i < nextLayer.size(); i++) {
      downstream += nextLayer.getNeuron(i).getWeight(hiddenIndex) *
                    nextLayer.getNeuron(i).getDelta();
    }

    neurons[hiddenIndex].computeHiddenDelta(downstream);
  }

public:
  Layer(int numNeurons, int inputSize) {
    for (int i = 0; i < numNeurons; i++) {
      neurons.emplace_back(inputSize);
    }
  }

  void computeAllHiddenDeltas(Layer &nextLayer) {
    for (int i = 0; i < neurons.size(); i++) {
      computeHiddenDeltas(nextLayer, i);
    }
  }

  vector<double> forward(const vector<double> &input) {
    vector<double> output;

    for (auto &neuron : neurons) {
      output.push_back(neuron.forward(input));
    }
    return output;
  }
  int size() const { return neurons.size(); }

  Neuron &getNeuron(int idx) { return neurons[idx]; }

  void computeOutputDeltas(const vector<double> &target) {
    for (int i = 0; i < neurons.size(); i++) {
      neurons[i].computeOutputDelta(target[i]);
    }
  }

  void updateWeights(double lr) {
    for (Neuron &neuron : neurons) {
      neuron.updateWeights(lr);
    }
  }
  const vector<Neuron> &getNeurons() const { return neurons; }
  vector<Neuron> &getNeurons() { return neurons; }
};

class NeuralNetwork {
private:
  Layer hidden;
  Layer output;
  vector<double> hiddenOut;
  vector<double> finalOut;

public:
  NeuralNetwork(int h, int o) : hidden(h, 9), output(9, o) {}

  vector<double> forward(const vector<double> &input) {
    hiddenOut = hidden.forward(input);
    finalOut = output.forward(hiddenOut);

    return finalOut;
  }

  void train(const vector<double> &input, const vector<double> &target,
             double lr) {
    forward(input);
    output.computeOutputDeltas(target);
    hidden.computeAllHiddenDeltas(output);
    output.updateWeights(lr);
    hidden.updateWeights(lr);
  }

  void saveWeights(const string &filename) {
    ofstream out(filename);

    for (const auto &neuron : hidden.getNeurons()) {
      for (double w : neuron.getWeights()) {
        out << w << ' ';
      }
      out << neuron.getBias() << '\n';
    }

    out << "OUTPUT \n";

    for (const auto &neuron : output.getNeurons()) {
      for (double w : neuron.getWeights()) {
        out << w << ' ';
      }
      out << neuron.getBias() << '\n';
    }

    out.close();
  }

  void loadWeights(const string &filename) {
    ifstream in(filename);

    if (!in) {
      throw runtime_error("Cannot open file");
    }

    vector<double> weights;
    double bias;

    for (Neuron &neuron : hidden.getNeurons()) {
      weights.resize(9);

      for (int i = 0; i < 9; i++) {
        in >> weights[i];
      }

      in >> bias;

      neuron.loadParameters(weights, bias);
    }

    string marker;
    in >> marker;

    for (Neuron &neuron : output.getNeurons()) {
      weights.resize(18);

      for (int i = 0; i < 18; i++) {
        in >> weights[i];
      }

      in >> bias;

      neuron.loadParameters(weights, bias);
    }

    in.close();
  }
};

int BestMove(vector<double> output, vector<double> board) {
  int best = -1;
  for (int i = 0; i < 9; i++) {
    if (board[i] != 0)
      continue;
    if (best == -1 || output[i] > output[best]) {
      best = i;
    }
  }
  return best;
}

double mse(const vector<double> &output, const vector<double> &target) {
  double loss = 0;
  for (int i = 0; i < 9; i++) {
    double err = target[i] - output[i];
    loss += err * err;
  }

  return loss / output.size();
}

#endif
