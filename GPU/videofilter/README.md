# OpenCL Video Filtering

Using the matrix multiplication library developed before, a video filtering algorithm was implemented.

## Results

| Frames | OpenCV Native | GPU      |
| ------ | ------------- | -------- |
| 10     | 44.18 FPS     | 6.13 FPS |
| 20     | 42.86 FPS     | 5.99 FPS |
| 50     | 41.98 FPS     | 5.81 FPS |
| 100    | 41.31 FPS     | 5.75 FPS |

The GPU code is much slower than the native OpenCV version. This is due to the inefficient way in which I am using it, without Task parallelization, with many buffer copy operations and implementing only the matrix multiplication step on GPU, not the whole convolution.