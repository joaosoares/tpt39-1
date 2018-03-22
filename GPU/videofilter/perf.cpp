#include "perf.hpp"

// perfStart returns the current time in milliseconds
std::chrono::high_resolution_clock::time_point perfStart() {
  return std::chrono::high_resolution_clock::now();
}

int perfDone(std::chrono::high_resolution_clock::time_point start) {
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}