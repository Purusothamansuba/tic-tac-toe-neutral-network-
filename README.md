# Tic-Tac-Toe Neural Network

A Tic-Tac-Toe AI implemented in C++ using a custom neural network trained on generated game data.

## Project Structure

```
.
├── NeuralNetwork.hpp
├── dataset.cpp
├── utils.cpp
└── xo.cpp
```

## Overview

This project explores the use of a custom neural network for Tic-Tac-Toe move selection. The network is trained on generated game data and later used to predict moves during gameplay.

The purpose of this repository is educational: learning neural network implementation, data generation, training workflows, and inference in modern C++.

## Build Instructions

### Step 1: Generate Dataset

**Clang++**
```bash
clang++ dataset.cpp -o dataset
./dataset
```

**G++**
```bash
g++ dataset.cpp -o dataset
./dataset
```

---

### Step 2: Train the Neural Network

**Clang++ (Recommended)**
```bash
clang++ utils.cpp \
  -O3 \
  -ffast-math \
  -march=native \
  -flto \
  -std=c++20 \
  -o utils
./utils
```

**G++**
```bash
g++ utils.cpp \
  -O3 \
  -ffast-math \
  -march=native \
  -flto \
  -std=c++20 \
  -o utils
./utils
```

---

### Step 3: Run Tic-Tac-Toe

**Clang++**
```bash
clang++ xo.cpp -std=c++20 -o xo
./xo
```

**G++**
```bash
g++ xo.cpp -std=c++20 -o xo
./xo
```

## Files

| File | Description |
|------|-------------|
| `NeuralNetwork.hpp` | Neural network architecture and implementation |
| `dataset.cpp` | Generates training data |
| `utils.cpp` | Trains the neural network |
| `xo.cpp` | Runs the Tic-Tac-Toe game using the trained model |

## Important Note

This project is **not** a perfect Tic-Tac-Toe solver and is not guaranteed to win every game.

The goal is to explore neural-network-based decision making rather than implement an optimal game-playing algorithm. Since the model learns from generated training data and does not use a complete search algorithm such as Minimax:

- The AI can make mistakes.
- The AI may lose against strong or optimal play.
- The AI may choose non-optimal moves.
- Results depend on the training data and training process.

This repository should be viewed as a machine learning experiment and educational project rather than a competitive Tic-Tac-Toe engine.

## Future Improvements

- Improved dataset generation
- Better neural network architectures
- Minimax + Neural Network hybrid approach
- Training metrics and evaluation tools
- Self-play training

## License

This project is provided for educational and learning purposes.
