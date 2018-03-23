#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <iostream>  // for standard I/O
#include "gpu.hpp"
#include "opencv2/opencv.hpp"
#include "perf.hpp"

using namespace cv;
using namespace std;
#define SHOW

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernel;

int main(int, char **) {
  // Initialize GPU
  gpuInitialize();
  // gpuShowInfo();
  VideoCapture camera("./bourne.mp4");
  if (!camera.isOpened())  // check if we succeeded
    return -1;

  const string NAME = "./output.avi";  // Form the new name with container
  int ex = static_cast<int>(CV_FOURCC('M', 'J', 'P', 'G'));
  Size S = Size((int)camera.get(CV_CAP_PROP_FRAME_WIDTH),  // Acquire input size
                (int)camera.get(CV_CAP_PROP_FRAME_HEIGHT));
  // Size S =Size(1280,720);
  cout << "SIZE:" << S << endl;

  VideoWriter outputVideo;  // Open the output
  outputVideo.open(NAME, ex, 25, S, true);

  if (!outputVideo.isOpened()) {
    cout << "Could not open the output video for write: " << NAME << endl;
    return -1;
  }
  int totalTime = 0;
  int count = 0;
  const char *windowName = "filter";  // Name shown in the GUI window.
#ifdef SHOW
  namedWindow(windowName);  // Resizable window, might not work on Windows.
#endif
  while (true) {
    Mat cameraFrame, displayframe;
    count = count + 1;
    if (count > 30) break;
    camera >> cameraFrame;
    Mat filterframe = Mat(cameraFrame.size(), CV_8UC3);
    Mat grayframe, edge_x, edge_y, edge, edge_inv;
    cvtColor(cameraFrame, grayframe, CV_BGR2GRAY);

    // CPU computation
    auto perf = perfStart();
    gpuGaussianBlur(grayframe, grayframe);
    gpuGaussianBlur(grayframe, grayframe);
    gpuGaussianBlur(grayframe, grayframe);
    // gpuGaussianBlur(grayframe, result);
    // result.copyTo(grayframe);
    // gpuGaussianBlur(grayframe, result);
    // result.copyTo(grayframe);
    // GaussianBlur(grayframe, grayframe, Size(3, 3), 0, 0);
    // GaussianBlur(grayframe, grayframe, Size(3, 3), 0, 0);
    // Scharr(grayframe, edge_x, CV_8U, 0, 1, 1, 0, BORDER_DEFAULT);
    // Scharr(grayframe, edge_y, CV_8U, 1, 0, 1, 0, BORDER_DEFAULT);
    // addWeighted(edge_x, 0.5, edge_y, 0.5, 0, edge);
    // threshold(edge, edge, 80, 255, THRESH_BINARY_INV);
    grayframe.copyTo(edge);
    auto perfResult = perfDone(perf);
    printf("CPU computation took %d milliseconds.\n", perfResult);

    cvtColor(edge, edge_inv, CV_GRAY2BGR);
    // Clear the output image to black, so that the cartoon line drawings will
    // be black (ie: not drawn).
    memset((char *)displayframe.data, 0, displayframe.step * displayframe.rows);
    grayframe.copyTo(displayframe, edge);
    cvtColor(displayframe, displayframe, CV_GRAY2BGR);
    outputVideo << displayframe;
#ifdef SHOW
    imshow(windowName, displayframe);
#endif
    totalTime += perfResult;
  }
  outputVideo.release();
  camera.release();
  printf("FPS %.2lf .\n", 299.0 / (totalTime / 1000.0));

  return EXIT_SUCCESS;
}
