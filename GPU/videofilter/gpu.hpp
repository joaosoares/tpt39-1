#ifndef GPU_HPP
#define GPU_HPP

#include "opencv2/opencv.hpp"
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <iostream>  // for standard I/O

using namespace cv;
using namespace std;

// gpuGaussianBlur applies a 3x3 gaussian blur on a float matrix
void gpuGaussianBlur(Mat matrix, Mat result);

// gpuCallback to use with openCL
void gpuCallback(const char *buffer, size_t length, size_t final, void *user_data);

int gpuInitialize();

#endif // GPU_HPP