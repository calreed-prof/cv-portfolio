# Edge Detection from Scratch

Implementated edge detection using hand-written convolution. no OpenCV. 
Sobel kernels are applied manually to detect horizontal and vertical gradients, which are 
then combined into a full edge map.

## Pipeline

1. Load image and convert RGB → Grayscale (manual weighted channel sum)
2. Apply Gaussian blur via hand-written convolution (3x3 kernel, σ=1)
3. Apply Sobel kernels in X and Y directions via manual convolution
4. Compute gradient magnitude (√(Gx² + Gy²))
5. Normalize and apply threshold to produce final edge map

## Topics
Convolution, Sobel kernels, gradients, partial derivatives, Gaussian blur, 
image preprocessing, NumPy

## Write-up
[Edge Detection from Scratch | Substack](https://open.substack.com/pub/calebkreed26/p/edge-detection-from-scratch?r=2ierv1&utm_campaign=post-expanded-share&utm_medium=post%20viewer)