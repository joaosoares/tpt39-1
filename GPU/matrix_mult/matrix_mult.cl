__kernel void matrix_mult(__global const float *A, __global const float *B,
                          __global float *X, int wA, int wB) {

  int tx = get_global_id(0);
  int ty = get_global_id(1);

  float curVal = 0;
  int i;
  for (i = 0; i < wA; i++) {
    curVal += A[tx * wA + i] * B[i * wB + ty];
  }
  X[tx * wB + ty] = curVal;
}
