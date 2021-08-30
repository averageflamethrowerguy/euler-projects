#include <iostream>
#include <map>

// rules of the lattice:
// 1. Arriving at the right forces you down
// 2. Arriving at the bottom forces you right
// 3. Arriving at the bottom right doesn't increase complexity.

// we will build in a "staircase pattern". Layer 0 is the bottom.
/*
 * X
 * X
 * X X X
 *
 * X
 * X X
 *   X X
 *
 * X
 * X X X
 *     X
 *
 * ...
 */

int width = 20;
int height = 20;
long grid[21][21];

long numberRoutes() {
    for (int i = 0; i < width; i++) {
        grid[height][i] = 1;
    }
    for (int j = 0; j < height; j++) {
        grid[j][width] = 1;
    }

    // pulls information to the upper left
    for (int i = width - 1; i >= 0; i--) {
        for (int j = height -1; j>= 0; j--) {
            grid[i][j] = grid[i+1][j] + grid[i][j+1];
        }
    }

    return grid[0][0];
}

int main() {
    std::cout << numberRoutes() << std::endl;
    return 0;
}
