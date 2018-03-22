#ifndef PERF_HPP
#define PERF_HPP

#include <chrono>

// perfStart returns the current time in milliseconds
std::chrono::high_resolution_clock::time_point perfStart();

// perfDone gets the perf object create before and returns the amount of milliseconds that have elapsed
int perfDone(std::chrono::high_resolution_clock::time_point start);

#endif // PERF_HPP