# Computational Diabetic Retinopathy Detection

This project explores feature extraction from retinal photographs for diabetic retinopathy (DR) classification. It provides:

- a MATLAB/CPU implementation for developing and visualizing the processing pipeline;
- a CUDA/GPU implementation for large-batch image preprocessing;
- feature matrices that can be used with MATLAB's Classification Learner or an SVM-based workflow.

The original CPU and GPU experiments were developed on Linux, but the source also targets macOS and other systems that provide compatible MATLAB/CUDA tooling.

## Project layout

```text
CPU/
  featureExtraction.m       MATLAB feature extraction pipeline
GPU/
  main.cpp                  CUDA preprocessing application
  ImageProcessing.cpp/h     CUDA image-processing operations
  Jpgd.cpp/h                JPEG decoder
  Jpge.cpp/h                JPEG encoder
  makefile                  Application and test build targets
  tests/
    ImageProcessingTest.cpp CUDA unit tests
```

## Requirements

### CPU/MATLAB

- MATLAB with the Image Processing Toolbox
- A folder containing retinal JPEG images

### GPU/CUDA

- An NVIDIA GPU and CUDA Toolkit
- `nvcc`
- Compute capability 3.5 or newer is used by the provided makefile

## Data format

Organize images by DR class:

```text
RetinalImages/
  0/
  1/
  2/
  3/
  4/
```

The GPU application expects a semicolon-separated `.txt` label file. Entries may use the format:

```text
10_left;10_right;11_left;11_right;30000_left
```

It combines each label with the resource directory and appends `.jpeg`, then processes matching files. The input is assumed to be grayscale-compatible; the original pipeline recommends the green channel.

## MATLAB/CPU feature extraction

Open `CPU/featureExtraction.m` in MATLAB and run:

```matlab
[MAT, rowValue] = featureExtraction( ...
    imagePath, class, destinationPath, plot);
```

Arguments:

- `imagePath`: directory containing retinal images;
- `class`: DR stage assigned to this dataset;
- `destinationPath`: directory for the generated feature table;
- `plot`: `true` to inspect results interactively, or `false` for unattended extraction.

The resulting table includes `Label`, `Class`, `Bloodvessel`, `Exudate`, `Haemorrhages`, and `Contrast` features.

## CUDA/GPU preprocessing

Build the application:

```bash
cd GPU
make
```

Run it:

```bash
./preProcessing
```

The program prompts for:

1. the DR stage;
2. the directory containing source images;
3. the destination directory;
4. the path to the semicolon-separated label file.

Processed RGB output is written to the destination directory, with the original image label as the filename.

## CUDA tests

The test executable checks image inversion, thresholding, subtraction, linear transformations, histogram operations, reductions, texture calculation, and Gaussian blur.

```bash
cd GPU
make test
```

Tests require a working CUDA device and `nvcc`.

To remove built executables:

```bash
make clean
```

## Classification

The CPU feature matrix can be imported into MATLAB's Classification Learner for model training and evaluation. Keep the image label and class with the extracted blood-vessel, exudate, haemorrhage, and contrast features.

Example feature header:

```text
Label,Class,Bloodvessel,Exudate,Haemorrhages,Contrast
```

## Dataset acknowledgment

This project used retinal image data from:

- Jorge Cuadros and George Bresnick, *Eyepacs: An adaptable telemedicine system for diabetic retinopathy screening*, Journal of Diabetes Science and Technology, 3(3):509-516, 2009.
- The Kaggle diabetic retinopathy detection dataset: https://www.kaggle.com/c/diabetic-retinopathy-detection

## License

MIT License. See [LICENSE.txt](LICENSE.txt).
