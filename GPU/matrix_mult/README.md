# OpenCL Matrix Multiplication

## Objectives

* Write a Matrix multiplication routine with two matrices of size M x K, K x N (where M=K=N)
* Measure speed up between CPU and GPU
* Use streamline to see various statistics about Cache/TLB miss.
* Measure Flops/S.

## Matrix multiplication code

## CPU vs GPU comparison

| M    | CPU (millis) | GPU time (millis) |
| ---- | ------------ | ----------------- |
| 128  | 14           | 2                 |
| 256  | 153          | 10                |
| 512  | 1181         | 75                |
| 1024 | 37305        | 535               |

## Streamline

## Performance analysis

To calculate an approximation of the number of floating point operations per second, let's consider the formula for total operations in the matrix multiplication:

$$ M _ N _ (2\*K - 1) $$ (1)

Since we took M = N = K, that is roughly

$$ 2 \* M^3 $$

So we build the table

| M   | Flops   | Flops/s (CPU) | Flops/s (GPU) |
| --- | ------- | ------------- | ------------- |
| 128 | 4194304 |

## References
