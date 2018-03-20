__kernel void vector_add(__global const float *x, 
                        __global const float *y, 
                        __global float *restrict z)
{
  z[get_global_id(0)] = x[get_global_id(0)] + y[get_global_id(0)];
}

