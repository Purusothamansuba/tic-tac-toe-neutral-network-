#include <fstream>
#include <iostream>
#include <map>
#include <vector>
using namespace std;

// 1 = X, -1 = O, 0 = empty
// X is always the one to move (we flip perspective when needed)

bool checkWin(const vector<int> &b, int player) {
  int p = player;
  return (b[0] == p && b[1] == p && b[2] == p) ||
         (b[3] == p && b[4] == p && b[5] == p) ||
         (b[6] == p && b[7] == p && b[8] == p) ||
         (b[0] == p && b[3] == p && b[6] == p) ||
         (b[1] == p && b[4] == p && b[7] == p) ||
         (b[2] == p && b[5] == p && b[8] == p) ||
         (b[0] == p && b[4] == p && b[8] == p) ||
         (b[2] == p && b[4] == p && b[6] == p);
}

bool isFull(const vector<int> &b) {
  for (int v : b)
    if (v == 0)
      return false;
  return true;
}

// Minimax: player=1 is maximizer, player=-1 is minimizer
int minimax(vector<int> &b, int player) {
  if (checkWin(b, 1))
    return 1;
  if (checkWin(b, -1))
    return -1;
  if (isFull(b))
    return 0;

  int best = -player; // worst for current player
  for (int i = 0; i < 9; i++) {
    if (b[i] != 0)
      continue;
    b[i] = player;
    int score = minimax(b, -player);
    b[i] = 0;
    if (player == 1)
      best = max(best, score);
    else
      best = min(best, score);
  }
  return best;
}

// Returns best move index for current player (always called with player=1
// perspective)
int bestMove(vector<int> &b, int player) {
  int best = -2;
  int move = -1;
  for (int i = 0; i < 9; i++) {
    if (b[i] != 0)
      continue;
    b[i] = player;
    int score = minimax(b, -player);
    b[i] = 0;
    if (score > best) {
      best = score;
      move = i;
    }
  }
  return move;
}

// Count pieces
int countPieces(const vector<int> &b, int p) {
  int c = 0;
  for (int v : b)
    if (v == p)
      c++;
  return c;
}

// Is board valid? X goes first so Xs >= Os and diff <= 1
bool isValidBoard(const vector<int> &b) {
  int xs = countPieces(b, 1);
  int os = countPieces(b, -1);
  return (xs == os || xs == os + 1);
}

// Whose turn is it?
int whoseTurn(const vector<int> &b) {
  int xs = countPieces(b, 1);
  int os = countPieces(b, -1);
  return (xs == os) ? 1 : -1; // X if equal, else O
}

// Game already over?
bool isTerminal(const vector<int> &b) {
  return checkWin(b, 1) || checkWin(b, -1) || isFull(b);
}

void saveCSV(const string &filename,
             const vector<pair<vector<int>, int>> &data) {
  ofstream file(filename);

  // Header
  for (int i = 0; i < 9; i++)
    file << "b" << i << ",";
  for (int i = 0; i < 9; i++) {
    file << "t" << i;
    if (i < 8)
      file << ",";
  }
  file << "\n";

  for (const auto &[board, move] : data) {
    // Board: store from X's perspective (neural net always sees X=1, O=-1)
    // If it's O's turn, we flip perspective so the mover is always 1
    int player = whoseTurn(board);
    for (int i = 0; i < 9; i++) {
      // flip if O's turn so "current player" = 1
      file << (board[i] * player);
      file << ",";
    }
    // Target
    for (int i = 0; i < 9; i++) {
      file << (i == move ? 1 : 0);
      if (i < 8)
        file << ",";
    }
    file << "\n";
  }
}

map<vector<int>, bool> visited;
vector<pair<vector<int>, int>> dataset;

void generate(vector<int> &board) {
  if (visited.count(board))
    return;
  visited[board] = true;

  if (!isValidBoard(board))
    return;
  if (isTerminal(board))
    return;

  int player = whoseTurn(board);
  int move = bestMove(board, player);

  dataset.push_back({board, move});

  // Recurse: try all legal moves
  for (int i = 0; i < 9; i++) {
    if (board[i] != 0)
      continue;
    board[i] = player;
    generate(board);
    board[i] = 0;
  }
}

int main() {
  vector<int> empty(9, 0);
  generate(empty);

  cout << "Total samples: " << dataset.size() << "\n";
  saveCSV("dataset.csv", dataset);
  cout << "Saved to dataset.csv\n";
  return 0;
}
