#include "gpu.hpp"

#define STRING_BUFFER_LEN 1024

// private non-exported function declarations
void matToConv(Mat imageMatrix, float *convMatrix, int rows, int cols);
void print_clbuild_errors(cl_program program, cl_device_id device);
unsigned char **read_file(const char *name);
void filter(Mat matrix, Mat result, float *kernel, int);
void checkError(int status, const char *msg);
float rand_float();
void matrixPrint(float *matrix, unsigned rows, unsigned cols);
void matrixMultiply(float *output, float *input_a, float *input_b, unsigned M,
                    unsigned N, unsigned K);

extern cl_platform_id platform;
extern cl_device_id device;
extern cl_context context;
extern cl_command_queue queue;
extern cl_program program;
extern cl_kernel kernel;

int gpuInitialize() {
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
  kernel = clCreateKernel(program, "matrix_mult", NULL);

  return 0;
}

// gpuGaussianBlur applies a 3x3 gaussian blur on a float matrix
void gpuGaussianBlur(Mat matrix, Mat result) {
  float kernel[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  filter(matrix, result, kernel, 1);
}

void filter(Mat matrix, Mat result, float *kernel, int numKernels) {
  const int kernelSize = 9;
  const int numElements = matrix.rows * matrix.cols;
  const int convMatrixSize = numElements * kernelSize;
  float convMatrix[convMatrixSize] = {0};
  float *output = (float *)malloc(numElements * sizeof(float));
  matrix.copyTo(result);

  // Convert matrix to do convolution
  matToConv(matrix, convMatrix, matrix.rows, matrix.cols);

  // Execute matrix multiplication
  matrixMultiply(output, convMatrix, kernel, numElements, numKernels,
                 kernelSize);
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
        }
        convMatrix[(i + j) * kernelSize + k] = curVal;
      }
    }
  }
}

void matrixMultiply(float *output, float *input_a, float *input_b, unsigned M,
                    unsigned N, unsigned K) {
  // Work sizes
  size_t localWorkSize[2], globalWorkSize[2];
  int status;

  localWorkSize[0] = 16;
  localWorkSize[1] = 16;
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
  status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize,
                                  localWorkSize, 2, write_event, &kernel_event);
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
  if (!*output) {
    fclose(fp);
    printf("mem allocate failure:%s", name);
    exit(-1);
  }

  if (!fread(*output, size, 1, fp)) printf("failed to read file\n");
  fclose(fp);
  printf("Output: %s", *output);
  return output;
}

void checkError(int status, const char *msg) {
  if (status != CL_SUCCESS) printf("%s\n", msg);
}

// Randomly generate a floating-point number between -10 and 10.
float rand_float() { return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f; }

// matrixPrint prints a formatted version of the matrix using printf
void matrixPrint(float *matrix, unsigned rows, unsigned cols) {
  for (unsigned i = 0; i < rows; i++) {
    printf("[");
    for (unsigned j = 0; j < cols; j++) {
      printf(" %7.2f ", matrix[i * cols + j]);
    }
    printf("]\n");
  }
}
