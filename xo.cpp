#include "NeuralNetwork.hpp"
#include <iostream>
#include <vector>

using namespace std;

void printBoard(const vector<int> &board) {
  cout << "\n";

  for (int i = 0; i < 9; i++) {
    char c = ' ';

    if (board[i] == 1)
      c = 'X';
    else if (board[i] == -1)
      c = 'O';
    else
      c = '0' + i;

    cout << " " << c << " ";

    if (i % 3 != 2)
      cout << "|";

    if (i % 3 == 2 && i != 8)
      cout << "\n-----------\n";
  }

  cout << "\n\n";
}

bool checkWin(const vector<int> &b, int p) {
  return (b[0] == p && b[1] == p && b[2] == p) ||
         (b[3] == p && b[4] == p && b[5] == p) ||
         (b[6] == p && b[7] == p && b[8] == p) ||
         (b[0] == p && b[3] == p && b[6] == p) ||
         (b[1] == p && b[4] == p && b[7] == p) ||
         (b[2] == p && b[5] == p && b[8] == p) ||
         (b[0] == p && b[4] == p && b[8] == p) ||
         (b[2] == p && b[4] == p && b[6] == p);
}

bool fullBoard(const vector<int> &b) {
  for (int x : b)
    if (x == 0)
      return false;

  return true;
}

int nnMove(NeuralNetwork &nn, const vector<int> &board) {
  vector<double> nnBoard(9);

  // NN is O (-1)
  // Convert to training perspective:
  // NN pieces -> 1
  // Human pieces -> -1

  for (int i = 0; i < 9; i++)
    nnBoard[i] = board[i] * (-1);

  vector<double> output = nn.forward(nnBoard);

  return BestMove(output, nnBoard);
}

int main() {
  int x = 128;
  NeuralNetwork nn(x, x);
  nn.loadWeights("model.nn");

  vector<int> board(9, 0);

  cout << "You are X\n";
  cout << "Cells:\n\n";

  printBoard(board);

  while (true) {
    int move;

    cout << "Your move (0-8): ";
    cin >> move;

    if (move < 0 || move > 8 || board[move] != 0) {
      cout << "Invalid move\n";
      continue;
    }

    board[move] = 1;

    printBoard(board);

    if (checkWin(board, 1)) {
      cout << "You win\n";
      break;
    }

    if (fullBoard(board)) {
      cout << "Draw\n";
      break;
    }

    int ai = nnMove(nn, board);

    cout << "NN plays: " << ai << "\n";

    board[ai] = -1;

    printBoard(board);

    if (checkWin(board, -1)) {
      cout << "NN wins\n";
      break;
    }

    if (fullBoard(board)) {
      cout << "Draw\n";
      break;
    }
  }

  return 0;
}
