# Comparing 3 Models on BDD100k Dataset

*Quick Note: We configured each model according to the best practices rather than keeping parameters constant, Because in real life, you would deploy the best version of each architecture*

### Models
- U-Net: 0.5914
- DeepLabV3+: 
- SegFormer: 

### Dataset: BDD100k
This dataset includes thousands of diverse dashcam images. This includes night, heavy weather, and much more. 
Although the dataset says 100k images, their segmentation dataset is much smaller at 10k images.

For each run I ran the same data pipeline:
- Removed 8 bad pairs, these images had differing H,W from input to label
- Read input image and converted to rgb
- Read label image with Grayscale
- Preloaded all pairs into memory (only about 37gb of memory at full 720p resolution)
- Ran albumentations:
    - Resize
    - Horizontal Flip
    - Color Jitter
    - Affine
    - Normalize
    - To Tensor

### Metric for Success
MIOU: Mean Intersection over Union

This metric takes the area of overlap divided by the area of union, or in simpler terms:

$$
\frac{\text{True Positives}}{\text{True Positives} + \text{False Positives} + \text{False Negatives}}
$$
![alt text](image.png)

# U-Net

- Parameters:
    - lr = 1e-3
    - encoder = 'efficientnet-b4'
    - encoder_weights = 'imagenet'
    - batch_size = 32
    - num_epochs = 50
    - patience = 7

- dtype = bfloat16
- Input Resolution = 360, 640
- Training GPU = A100
- Loss = Focal + Dice
    - **Focal Loss**: This loss function is for extreme class imbalances. in BDD100k there are multiple classes that barely take up any image space. This function works perfectly to combat that.
    - **Dice Loss**: This loss function is also perfect for class imbalance. Additionally, it works directly in tandem with MIOU helping us optimize for our metric. 
- Optimizer = AdamW
- Scheduler = Cosine Annealing
    - Specifically works great with U-Net

## Results

| MIOU   | MIOU (Train Excl) | Iterations | Params | MACs  | Latency | FPS  | Size   |
|--------|-------------------|------------|--------|-------|---------|------|--------|
| 59.14% | 62.43%            | ~11k        | 2.8M   | 11.9G | 22.78ms | 43.9 | 81.9MB |

*Quick Latency Notes: A100, 360x640, bfloat16, batch size 1*

Training Loss:
![](media/W&B%20Chart%206_22_2026,%203_04_59%20PM.png)

We had a val loss hiccup on iteration 7, this is attributed to the Cosine Scheduler

Per Class IOU:

![alt text](media/unet-per-class-iou.png)

The model struggles with classes that do not make up a lot of the image. Even with focal loss there is simply not enough data to get high accuracy in low frequency classes. The train class occurs basically zero times within the data which can be seen by the 0 IOU score.

Example Output:

![alt text](media/unet-example.png)

This is impressive for an architecture that came out 11 years ago.

# DeepLabV3+

Work in Progress