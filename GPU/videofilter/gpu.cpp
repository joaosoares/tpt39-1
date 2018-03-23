#include "gpu.hpp"

#define STRING_BUFFER_LEN 1024

// private non-exported function declarations
void matToConv(Mat imageMatrix, float *convMatrix, int rows, int cols);
void convToMat(float *convMatrix, Mat result, int rows, int cols);
void print_clbuild_errors(cl_program program, cl_device_id device);
unsigned char **read_file(const char *name);
void filter(Mat matrix, Mat result, float *kernel, int);
void checkError(int status, const char *msg);
float rand_float();
void matrixPrint(float *matrix, unsigned rows, unsigned cols);
void matrixMultiply(float *output, float *input_a, float *input_b, unsigned M,
                    unsigned N, unsigned K);

// cpuMatrixMultiply uses the CPU to multiply two input matrices with proper
// dimensions
void cpuMatrixMultiply(float *X, float *A, float *B, unsigned dim1,
                       unsigned dim2, unsigned dimShared) {
  for (unsigned i = 0; i < dim1; i++) {
    for (unsigned j = 0; j < dim2; j++) {
      for (unsigned k = 0; k < dimShared; k++) {
        X[i * dim2 + j] += A[i * dimShared + k] * B[k * dim2 + j];
      }
    }
  }
}

extern cl_platform_id platform;
extern cl_device_id device;
extern cl_context context;
extern cl_command_queue queue;
extern cl_program program;
extern cl_kernel kernel;

int gpuInitialize() {
  int status = 0;
  char char_buffer[STRING_BUFFER_LEN];

  cl_context_properties context_properties[] = {
      CL_CONTEXT_PLATFORM,
      0,
      CL_PRINTF_CALLBACK_ARM,
      (cl_context_properties)gpuCallback,
      CL_PRINTF_BUFFERSIZE_ARM,
      0x1000,
      0};

  clGetPlatformIDs(1, &platform, NULL);

  clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer,
                    NULL);
  printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
  clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN,
                    char_buffer, NULL);
  printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
  clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN,
                    char_buffer, NULL);
  printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);

  context_properties[1] = (cl_context_properties)platform;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
  queue = clCreateCommandQueue(context, device, 0, NULL);

  // Program compilation
  unsigned char **opencl_program = read_file("matrix_mult.cl");
  program = clCreateProgramWithSource(context, 1, (const char **)opencl_program,
                                      NULL, NULL);
  if (program == NULL) {
    printf("Program creation failed\n");
    return -1;
  }

  int success = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (success != CL_SUCCESS) print_clbuild_errors(program, device);
  kernel = clCreateKernel(program, "matrix_mult", &status);

  printf("Error code for kernel creation: %d\n", status);

  return 0;
}

// gpuGaussianBlur applies a 3x3 gaussian blur on a float matrix
void gpuGaussianBlur(Mat matrix, Mat result) {
  printf("Starting gpuGaussianBlur\n");
  float kernel[9] = {0.077847, 0.123317, 0.077847, 0.123317, 0.195346,
                     0.123317, 0.077847, 0.123317, 0.077847};
  filter(matrix, result, kernel, 1);
}

// gpuSobelHorizontal applies a 3x3 Sobel / Scharr filter on the x axis
void gpuSobelHorizontal(Mat matrix, Mat result) {
  printf("Starting gpuSobelHorizontal\n");
  float kernel[9] = {3, 0, -3, 10, 0, -10, 3, 0, -3};
  filter(matrix, result, kernel, 1);
}

// gpuSobelVertical applies a 3x3 Sobel / Scharr filter on the y axis
void gpuSobelVertical(Mat matrix, Mat result) {
  printf("Starting gpuSobelVertical\n");
  float kernel[9] = {3, 10, 3, 0, 0, 0, -3, -10, -3};
  filter(matrix, result, kernel, 1);
}

void filter(Mat matrix, Mat result, float *kernel, int numKernels) {
  const int kernelSize = 9;
  const int numElements = matrix.rows * matrix.cols;
  const int convMatrixSize = numElements * kernelSize;
  float convMatrix[convMatrixSize] = {0};
  float *output = (float *)malloc(numElements * sizeof(float));
  matrix.convertTo(matrix, CV_32FC1);
  Mat temp_result = Mat(matrix.size(), CV_32FC1, 1.f / 255);
  // matrix.copyTo(result);

  // printf("Kernel:\n");
  // matrixPrint(kernel, 9, 1);

  // printf("First pixels for input matrix: \n");
  // printf("[ %7.2f ]\n", matrix.at<float>(0, 0));
  // printf("[ %7.2f ]\n", matrix.at<float>(0, 1));
  // printf("[ %7.2f ]\n", matrix.at<float>(0, 2));
  // printf("[ %7.2f ]\n", matrix.at<float>(0, 3));
  // printf("[ %7.2f ]\n", matrix.at<float>(0, 4));
  // printf("[ %7.2f ]\n", matrix.at<float>(0, 5));

  // printf("Starting matToConv\n");
  // Convert matrix to do convolution
  matToConv(matrix, convMatrix, matrix.rows, matrix.cols);
  // matrixPrint(convMatrix, matrix.rows * matrix.cols, 9);

  // printf("Starting matrixMultiply\n");
  // Execute matrix multiplication
  matrixMultiply(output, convMatrix, kernel, numElements, numKernels,
                 kernelSize);

  // printf("Mult result:\n");
  // matrixPrint(output, numElements, 1);

  // printf("Starting convToMat\n");
  convToMat(output, temp_result, matrix.rows, matrix.cols);

  // printf("First pixels for result matrix (float): \n");
  // printf("[ %7.2f ]\n", temp_result.at<float>(0, 0));
  // printf("[ %7.2f ]\n", temp_result.at<float>(0, 1));
  // printf("[ %7.2f ]\n", temp_result.at<float>(0, 2));
  // printf("[ %7.2f ]\n", temp_result.at<float>(0, 3));
  // printf("[ %7.2f ]\n", temp_result.at<float>(0, 4));
  // printf("[ %7.2f ]\n", temp_result.at<float>(0, 5));

  // gpuFloatMatPrint(temp_result);

  temp_result.convertTo(temp_result, CV_8U);

  temp_result.copyTo(result);
  // printf("First pixels for result matrix (integer): \n");
  // printf("[ %3d ]\n", result.at<int>(0, 0));
  // printf("[ %3d ]\n", result.at<int>(0, 1));
  // printf("[ %3d ]\n", result.at<int>(0, 2));
  // printf("[ %3d ]\n", result.at<int>(0, 3));
  // printf("[ %3d ]\n", result.at<int>(0, 4));
  // printf("[ %3d ]\n", result.at<int>(0, 5));
}

// Given an OpenCV Mat, create a float matrix to do a convolution where each
// element of the original element creates a row of kernelSize elements.
// kernelSize is fixed at 3x3
void matToConv(Mat imageMatrix, float *convMatrix, int rows, int cols) {
  float curVal;
  int curIndX, curIndY;
  const int kernelSize = 9;
  const int KERNEL_MAP_X[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
  const int KERNEL_MAP_Y[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};

  int curIdx = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      for (int k = 0; k < kernelSize; k++) {
        curIndX = i + KERNEL_MAP_X[k];
        curIndY = j + KERNEL_MAP_Y[k];
        if ((curIndX < 0) || (curIndX >= rows) || (curIndY < 0) ||
            (curIndY >= cols)) {
          curVal = 0;
        } else {
          curVal = imageMatrix.at<float>(curIndX, curIndY);
          // if ((i < 5) && (j < 5)) {
          //   printf("Curval: %.2f\n", curVal);
          // }
        }
        convMatrix[curIdx * kernelSize + k] = curVal;
      }
      curIdx++;
    }
  }
}

void convToMat(float *convMatrix, Mat result, int rows, int cols) {
  int curIdx = 0;

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      result.at<float>(i, j) = convMatrix[curIdx];
      // if (curIdx < 5) {
      //   printf("[ %7.2f ]\n", convMatrix[curIdx]);
      //   printf("[ %7.2f ]\n", result.at<float>(i, j));
      // }
      curIdx++;
    }
  }
}

void matrixMultiply(float *output, float *input_a, float *input_b, unsigned M,
                    unsigned N, unsigned K) {
  // Work sizes
  // size_t localWorkSize[2];
  size_t globalWorkSize[2];
  int status;

  // localWorkSize[0] = 10;
  // localWorkSize[1] = 10;
  globalWorkSize[0] = M;
  globalWorkSize[1] = N;

  // Input buffers.
  cl_mem bufferInputA = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                       M * K * sizeof(float), NULL, &status);
  checkError(status, "Failed to create buffer for input A");

  cl_mem bufferInputB = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                       K * N * sizeof(float), NULL, &status);
  checkError(status, "Failed to create buffer for input A");

  // Output buffer.
  cl_mem bufferOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                       M * N * sizeof(float), NULL, &status);
  checkError(status, "Failed to create buffer for output");

  // Transfer inputs to each device. Each of the host buffers supplied to
  // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
  // for the host-to-device transfer.
  cl_event write_event[2];
  cl_event kernel_event, finish_event;
  status = clEnqueueWriteBuffer(queue, bufferInputA, CL_FALSE, 0,
                                M * K * sizeof(float), input_a, 0, NULL,
                                &write_event[0]);
  checkError(status, "Failed to transfer input A");

  status = clEnqueueWriteBuffer(queue, bufferInputB, CL_FALSE, 0,
                                K * N * sizeof(float), input_b, 0, NULL,
                                &write_event[1]);
  checkError(status, "Failed to transfer input B");

  // Set kernel arguments.
  unsigned argi = 0;

  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &bufferInputA);
  checkError(status, "Failed to set argument 1");

  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &bufferInputB);
  checkError(status, "Failed to set argument 2");

  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &bufferOutput);
  checkError(status, "Failed to set argument 3");

  status = clSetKernelArg(kernel, argi++, sizeof(int), &K);
  checkError(status, "Failed to set argument 4");

  status = clSetKernelArg(kernel, argi++, sizeof(int), &N);
  checkError(status, "Failed to set argument 5");

  // Enqueue as many kernels as it fits on the machine
  status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, NULL,
                                  2, write_event, &kernel_event);
  checkError(status, "Failed to launch kernel");

  // // Read the result. This the final operation.
  status = clEnqueueReadBuffer(queue, bufferOutput, CL_TRUE, 0,
                               M * N * sizeof(float), output, 1, &kernel_event,
                               &finish_event);

  // Release local events.
  clReleaseEvent(write_event[0]);
  clReleaseEvent(write_event[1]);
  // clReleaseKernel(kernel);
  // clReleaseCommandQueue(queue);
  clReleaseMemObject(bufferInputA);
  clReleaseMemObject(bufferInputB);
  clReleaseMemObject(bufferOutput);
  // clReleaseProgram(program);
  // clReleaseContext(context);

  // //--------------------------------------------------------------------

  // clFinish(queue);

  return;
}

void gpuCallback(const char *buffer, size_t length, size_t final,
                 void *user_data) {
  fwrite(buffer, 1, length, stdout);
}

void print_clbuild_errors(cl_program program, cl_device_id device) {
  cout << "Program Build failed\n";
  size_t length;
  char buffer[2048];
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer),
                        buffer, &length);
  cout << "--- Build log ---\n " << buffer << endl;
  exit(1);
}

unsigned char **read_file(const char *name) {
  size_t size;
  unsigned char **output = (unsigned char **)malloc(sizeof(unsigned char *));
  FILE *fp = fopen(name, "rb");
  if (!fp) {
    printf("no such file:%s", name);
    exit(-1);
  }

  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  *output = (unsigned char *)malloc(size);
  unsigned char **outputstr = (unsigned char **)malloc(sizeof(unsigned char *));
  *outputstr = (unsigned char *)malloc(size);
  if (!*output) {
    fclose(fp);
    printf("mem allocate failure:%s", name);
    exit(-1);
  }

  if (!fread(*output, size, 1, fp)) printf("failed to read file\n");
  fclose(fp);
  printf("file size %d\n", size);
  printf("-------------------------------------------\n");
  snprintf((char *)*outputstr, size, "%s\n", *output);
  printf("%s\n", *outputstr);
  printf("-------------------------------------------\n");
  return outputstr;
}

void checkError(int status, const char *msg) {
  if (status != CL_SUCCESS) printf("%s\n", msg);
}

// Randomly generate a floating-point number between -10 and 10.
float rand_float() { return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f; }

// matrixPrint prints a formatted version of the matrix using printf
void matrixPrint(float *matrix, unsigned rows, unsigned cols) {
  for (unsigned i = 0; i < rows; i++) {
    if ((i < 6) || (i > rows - 5)) {
      printf("[");
      for (unsigned j = 0; j < cols; j++) {
        if ((i == 5) || (j == 5)) {
          printf("    ...  ");
        } else if ((j < 5) || (j > cols - 5)) {
          printf(" %7.2f ", matrix[i * cols + j]);
        }
      }
      printf("]\n");
    }
  }
}

// gpuFloatMatPrint prints a formatted version of the matrix using printf
void gpuFloatMatPrint(Mat matrix) {
  unsigned rows = matrix.rows;
  unsigned cols = matrix.cols;
  float value;
  for (unsigned i = 0; i < rows; i++) {
    if ((i < 6) || (i > rows - 5)) {
      printf("[");
      for (unsigned j = 0; j < cols; j++) {
        if (((i == 5) && (j < 6)) || (j == 5)) {
          printf("    ...  ");
        } else if ((j < 5) || (j > cols - 5)) {
          value = matrix.at<float>(i, j);
          printf(" %7.2f ", value);
        }
      }
      printf("]\n");
    }
  }
}

// gpuIntMatPrint prints a formatted version of the matrix using printf
void gpuIntMatPrint(Mat matrix) {
  unsigned rows = matrix.rows;
  unsigned cols = matrix.cols;
  int value;
  for (unsigned i = 0; i < rows; i++) {
    if ((i < 6) || (i > rows - 5)) {
      printf("[");
      for (unsigned j = 0; j < cols; j++) {
        if (((i == 5) && (j < 6)) || (j == 5)) {
          printf("    ...  ");
        } else if ((j < 5) || (j > cols - 5)) {
          value = matrix.at<int>(i, j);
          printf(" %3d ", value);
        }
      }
      printf("]\n");
    }
  }
}
