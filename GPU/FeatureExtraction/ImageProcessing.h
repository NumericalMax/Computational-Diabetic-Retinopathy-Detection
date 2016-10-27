#include <stdint.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_device_runtime_api.h>
#include <sm_20_atomic_functions.h>
#include <sm_20_intrinsics.h>

using namespace std;

class ImageProcessing{

public:

	ImageProcessing();
	~ImageProcessing();

	//HISTOGRAM
	void histogramGPU(uint8_t *d_Data, unsigned int *h_Result, int width, int height, bool timing);
	void histogramEqualizationGPU(uint8_t *currentImage, uint8_t *outputImage, unsigned int *histTrafo, int width, int height, bool timing);
	
	//MORPHOLOGICAL OPERATIONS
	void dilateGPU(uint8_t *currentImage, uint8_t *outputImage, int structure, int width, int height, bool timing);
	void erodeGPU(uint8_t *currentImage, uint8_t *outputImage, int structure, int width, int height, bool timing);
	
	//KERNEL FILTER
	void medianGPU(uint8_t *currentImage, uint8_t *outputImage, int radius, int width, int height, bool timing);
	void sobelGPU(uint8_t *currentImage, uint8_t *outputImage, int width, int height, int limit, bool timing);
	void gaussianBlurGPU(uint8_t *currentImage, uint8_t *outputImage, int width, int height, int radius, bool timing);

	//REDUCTION OPERATIONS
	void maximalElementGPU(uint8_t *currentImage, int &val, int &x, int &y, int width, int height, bool timing);
	void getAreaGPU(uint8_t *currentImage, double &area, int width, int height, bool timing);
	void meanVaueGPU(uint8_t *currentImage, int &val, int width, int height, bool timing);

	//NAIV IMAGE PROCESSING TECHNIQUES
	void invertGPU(uint8_t *currentImage, uint8_t *outputImage, int width, int height, bool timing);
	void blackwhiteGPU(uint8_t *currentImage, uint8_t *outputImage, int threshold, int width, int height, bool timing);
	void subtrachtImagesGPU(uint8_t *inImage1, uint8_t *inImage2, uint8_t *outputImage, int width, int height, bool timing);
	void imadjustGPU(uint8_t *currentImage, uint8_t *outputImage, int width, int height, int lowerLimit, int upperLimit, double lowerScale, double upperScale, bool timing);
	void removeAreaGPU(uint8_t *currentImage, uint8_t *outputImage, int width, int height, int x, int y, int radius, bool timing);
	void linearTransformationGPU(uint8_t *currentImage, uint8_t *outputImage, int width, int height, float alpha, int beta, bool timing);

	//CO OCCURENCE MATRIX
	void textureGPU(uint8_t *currentImage, double &contrast, int width, int height, bool timing);

};
