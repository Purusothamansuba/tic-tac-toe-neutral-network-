#include "NeuralNetwork.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

using namespace std;

struct TrainingSample {
  vector<double> board;
  vector<double> target;
};

vector<TrainingSample> loadCSV(const string &filename) {

  vector<TrainingSample> data;

  ifstream file(filename);

  string line;

  getline(file, line);

  while (getline(file, line)) {

    stringstream ss(line);

    string token;
    vector<double> vals;

    while (getline(ss, token, ',')) {
      vals.push_back(stod(token));
    }

    TrainingSample s;

    s.board = vector<double>(vals.begin(), vals.begin() + 9);

    s.target = vector<double>(vals.begin() + 9, vals.end());

    data.push_back(s);
  }

  return data;
}

void saveCSV(const string &filename, const vector<TrainingSample> &data) {

  ofstream file(filename);

  for (int i = 0; i < 9; i++)
    file << "b" << i << ",";

  for (int i = 0; i < 9; i++) {

    file << "t" << i;

    if (i < 8)
      file << ",";
  }

  file << '\n';

  for (const auto &s : data) {

    for (double v : s.board)
      file << v << ",";

    for (int i = 0; i < (int)s.target.size(); i++) {

      file << s.target[i];

      if (i < (int)s.target.size() - 1)
        file << ",";
    }

    file << '\n';
  }
}

void evaluate(NeuralNetwork &nn, const vector<TrainingSample> &dataset) {

  int correct = 0;

  for (const auto &s : dataset) {

    vector<double> output = nn.forward(s.board);

    int predicted = BestMove(output, s.board);

    int expected = -1;

    for (int i = 0; i < 9; i++) {

      if (s.target[i] == 1.0) {
        expected = i;
        break;
      }
    }

    if (predicted == expected)
      correct++;
  }

  cout << "\n========================\n";
  cout << "Samples  : " << dataset.size() << '\n';
  cout << "Correct  : " << correct << '\n';
  cout << "Wrong    : " << dataset.size() - correct << '\n';
  cout << "Accuracy : " << (100.0 * correct / dataset.size()) << "%\n";
  cout << "========================\n";
}

void train() {

  int x = 128;
  NeuralNetwork nn(x, x);

  vector<TrainingSample> dataset = loadCSV("dataset.csv");

  cout << "Dataset size: " << dataset.size() << '\n';

  mt19937 rng(random_device{}());

  for (int epoch = 0; epoch < 10000; epoch++) {

    shuffle(dataset.begin(), dataset.end(), rng);

    for (const auto &s : dataset) {

      nn.train(s.board, s.target, 0.1);
    }

    if (epoch % 100 == 0) {

      double total = 0;

      for (const auto &s : dataset) {

        total += mse(nn.forward(s.board), s.target);
      }

      cout << "Epoch " << epoch << " Loss: " << total / dataset.size() << '\n';
    }
  }

  evaluate(nn, dataset);

  nn.saveWeights("model.nn");
}

int main() {
  train();
  return 0;
}
