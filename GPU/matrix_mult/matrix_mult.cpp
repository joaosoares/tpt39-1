#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <iostream>  // for standard I/O
#define STRING_BUFFER_LEN 1024
using namespace std;

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
  return output;
}

void callback(const char *buffer, size_t length, size_t final,
              void *user_data) {
  fwrite(buffer, 1, length, stdout);
}

void checkError(int status, const char *msg) {
  if (status != CL_SUCCESS) printf("%s\n", msg);
}

// Randomly generate a floating-point number between -10 and 10.
float rand_float() { return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f; }

// perfStart returns the current time in milliseconds
std::chrono::high_resolution_clock::time_point perfStart() {
  return std::chrono::high_resolution_clock::now();
}

int perfDone(std::chrono::high_resolution_clock::time_point start) {
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

// matrixPopulateRand fills a given matrix with random float values.
void matrixPopulateRand(float *matrix, unsigned rows, unsigned cols) {
  for (unsigned i = 0; i < rows; i++) {
    for (unsigned j = 0; j < cols; j++) {
      matrix[i * cols + j] = rand_float();
    }
  }
}

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

// matrixMultiply multiplies two input matrices with proper dimensions
void matrixMultiply(float *A, float *B, float *X, unsigned dim1,
                    unsigned dimShared, unsigned dim2) {
  for (unsigned i = 0; i < dim1; i++) {
    for (unsigned j = 0; j < dim2; j++) {
      for (unsigned k = 0; k < dimShared; k++) {
        X[i * dim2 + j] += A[i * dimShared + k] * B[k * dim2 + j];
      }
    }
  }
}

int main() {
  char char_buffer[STRING_BUFFER_LEN];
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_context_properties context_properties[] = {CL_CONTEXT_PLATFORM,
                                                0,
                                                CL_PRINTF_CALLBACK_ARM,
                                                (cl_context_properties)callback,
                                                CL_PRINTF_BUFFERSIZE_ARM,
                                                0x1000,
                                                0};
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;

  //--------------------------------------------------------------------

  // Sizes for the matrices
  const unsigned M = 128;
  const unsigned N = 256;
  const unsigned K = 512;

  // Allocate memory for first matrix
  float *input_a = (float *)malloc(sizeof(float) * M * K);

  // Allocate memory for second matrix
  float *input_b = (float *)malloc(sizeof(float) * K * N);

  // Allocate memory for result matrix
  float *output = (float *)malloc(sizeof(float) * M * N);

  // Allocate memory for reference matrix
  float *reference = (float *)malloc(sizeof(float) * M * N);

  // OpenCL buffers
  cl_mem bufferInputA;  // num_devices elements
  cl_mem bufferInputB;  // num_devices elements
  cl_mem bufferOutput;  // num_devices elements

  // Work sizes
  size_t localWorkSize[2], globalWorkSize[2];

  localWorkSize[0] = 16;
  localWorkSize[1] = 16;
  globalWorkSize[0] = M;
  globalWorkSize[1] = N;

  // Populate input matrices with random values
  matrixPopulateRand(input_a, M, K);
  matrixPopulateRand(input_b, K, N);

  // Print matrices to check correctness
  // printf("Matrix A:\n");
  // matrixPrint(input_a, M, K);
  // printf("Matrix B:\n");
  // matrixPrint(input_b, K, N);

  // Execute CPU matrix multiplication
  auto perf = perfStart();
  matrixMultiply(input_a, input_b, reference, M, K, N);
  auto perfResult = perfDone(perf);
  printf("CPU computation took %d milliseconds.\n", perfResult);

  // Print result of multiplication
  // printf("Expected A * B:\n");
  // matrixPrint(reference, M, N);

  // Initialize GPU
  int status;

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
    return 1;
  }
  int success = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (success != CL_SUCCESS) print_clbuild_errors(program, device);
  kernel = clCreateKernel(program, "matrix_mult", NULL);

  // Input buffers.
  bufferInputA = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                M * K * sizeof(float), NULL, &status);
  checkError(status, "Failed to create buffer for input A");

  bufferInputB = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                K * N * sizeof(float), NULL, &status);
  checkError(status, "Failed to create buffer for input A");

  // Output buffer.
  bufferOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
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
  perf = perfStart();
  status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize,
                                  localWorkSize, 2, write_event, &kernel_event);
  checkError(status, "Failed to launch kernel");

  // // Read the result. This the final operation.
  status = clEnqueueReadBuffer(queue, bufferOutput, CL_TRUE, 0,
                               M * N * sizeof(float), output, 1, &kernel_event,
                               &finish_event);
  perfResult = perfDone(perf);
  printf("GPU computation took %d milliseconds.\n", perfResult);

  // time(&end);
  // diff = difftime(end, start);
  // printf("GPU took %.8lf seconds to run.\n", diff);
  // // Verify results.

  // Print result of multiplication
  // printf("Actual A * B:\n");
  // matrixPrint(output, M, N);

  float diff = 0.0;
  float actual = 0.0;
  float expected = 0.0;
  for (unsigned i = 0; i < M; i++) {
    for (unsigned j = 0; j < N; j++) {
      actual = output[i * N + j];
      expected = reference[i * N + j];
      diff = expected - actual;
      if (fabsf(diff) > 1.0e-5f) {
        printf(
            "Failed verification @ index (%d, %d) \nExpected: %f \nActual: "
            "%f\n",
            i, j, expected, actual);
        return 1;
      }
    }
  }

  // Release local events.
  clReleaseEvent(write_event[0]);
  clReleaseEvent(write_event[1]);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseMemObject(bufferInputA);
  clReleaseMemObject(bufferInputB);
  clReleaseMemObject(bufferOutput);
  clReleaseProgram(program);
  clReleaseContext(context);

  // //--------------------------------------------------------------------

  // clFinish(queue);

  return 0;
}
