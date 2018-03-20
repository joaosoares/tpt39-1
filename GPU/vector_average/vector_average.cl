__kernel void vector_average(__global const float *x, __global float *restrict z)
{
  printf("z before %f\n", *z);
  *z += x[get_global_id(0)];/// get_global_size(0);
  printf("cur %f\n", *z);
}