__kernel void vector_average(__global const float *x, __global float *restrict z)
{
  int i;
  for (i = 0; i < get_local_size(0); i++) { // size of input buffer
    z[get_group_id(0)] += x[i];
  }
}