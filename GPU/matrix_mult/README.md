# OpenCL Matrix Multiplication

## Objectives

* Write a Matrix multiplication routine with two matrices of size M x K, K x N (where M=K=N)
* Measure speed up between CPU and GPU
* Measure Flops/S.

## Matrix multiplication code

## CPU vs GPU comparison

| M    | CPU (millis) | GPU time (millis) |
| ---- | ------------ | ----------------- |
| 128  | 14           | 2                 |
| 256  | 153          | 10                |
| 512  | 1181         | 75                |
| 1024 | 37305        | 535               |

## Performance analysis

To calculate an approximation of the number of floating point operations per second, let's consider the formula for total operations in the matrix multiplication:

$$ M * N * (2*K - 1) $$

Since we took M = N = K, that is roughly

$$ 2 * M^3 $$

So we build the table

| M    | Flops         | Flops/s (CPU) | Flops/s (GPU) |
| ---- | ------------- | ------------- | ------------- |
| 128  | 4,194,304     | 300 MFlops    | 2.10 GFlops   |
| 256  | 33,554,432    | 219 MFlops    | 3.29 GFlops   |
| 512  | 268,435,456   | 227 MFlops    | 3.58 GFlops   |
| 1024 | 2,147,483,648 | 57.6 MFlops   | 4.01 GFlops   |

As we can see, CPU decreases performance, probably because memory access becomes more expensive due to the cache not fitting the bigger sets of data. Meanwhile, the GPU becomes actually faster, maybe due to better use of its resources for the operations, as our naive implementations leaves many gaps in resource usage.

## References
* http://www.es.ele.tue.nl/~mwijtvliet/5KK73/?page=mmopencl (by Technische Universiteit Eindhoven).

