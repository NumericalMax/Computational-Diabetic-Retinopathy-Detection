#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../ImageProcessing.h"

#define CUDA_CHECK(call)                                                        \
	do {                                                                        \
		cudaError_t error = (call);                                             \
		if (error != cudaSuccess) {                                             \
			std::fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
				cudaGetErrorString(error), __FILE__, __LINE__);                 \
			std::exit(EXIT_FAILURE);                                            \
		}                                                                       \
	} while (0)

static int failures = 0;

#define TEST(name) static void name()
#define RUN_TEST(name) do { \
	std::printf("  %-32s", #name); \
	name(); \
	std::printf("PASS\n"); \
} while (0)

static void require(bool condition, const char *message)
{
	if (!condition) {
		std::fprintf(stderr, "\nFAIL: %s\n", message);
		failures++;
	}
}

static void requirePixel(const std::vector<uint8_t> &actual,
	const std::vector<uint8_t> &expected, const char *message)
{
	require(actual.size() == expected.size(), message);
	if (actual.size() != expected.size())
		return;
	for (size_t i = 0; i < actual.size(); i++) {
		if (actual[i] != expected[i]) {
			std::fprintf(stderr, "\nFAIL: %s at pixel %zu: got %u, expected %u\n",
				message, i, static_cast<unsigned>(actual[i]),
				static_cast<unsigned>(expected[i]));
			failures++;
			return;
		}
	}
}

static uint8_t *copyToDevice(const std::vector<uint8_t> &host)
{
	uint8_t *device = NULL;
	CUDA_CHECK(cudaMalloc(&device, host.size()));
	CUDA_CHECK(cudaMemcpy(device, host.data(), host.size(), cudaMemcpyHostToDevice));
	return device;
}

static std::vector<uint8_t> copyFromDevice(uint8_t *device, size_t size)
{
	std::vector<uint8_t> host(size);
	CUDA_CHECK(cudaMemcpy(host.data(), device, size, cudaMemcpyDeviceToHost));
	return host;
}

TEST(testInvert)
{
	const int width = 5;
	const int height = 3;
	std::vector<uint8_t> input = {0, 10, 64, 128, 255,
		1, 2, 200, 254, 100,
		9, 19, 99, 199, 249};
	uint8_t *deviceInput = copyToDevice(input);
	uint8_t *deviceOutput = NULL;
	CUDA_CHECK(cudaMalloc(&deviceOutput, input.size()));

	ImageProcessing processing;
	processing.invertGPU(deviceInput, deviceOutput, width, height, false);
	CUDA_CHECK(cudaThreadSynchronize());

	std::vector<uint8_t> output = copyFromDevice(deviceOutput, input.size());
	std::vector<uint8_t> expected;
	for (uint8_t value : input)
		expected.push_back(static_cast<uint8_t>(255 - value));

	requirePixel(output, expected, "invertGPU");
	CUDA_CHECK(cudaFree(deviceInput));
	CUDA_CHECK(cudaFree(deviceOutput));
}

TEST(testBlackWhiteAndSubtract)
{
	const int width = 3;
	const int height = 2;
	std::vector<uint8_t> input = {10, 127, 128, 200, 255, 0};
	std::vector<uint8_t> second = {20, 100, 30, 250, 10, 5};
	uint8_t *deviceInput = copyToDevice(input);
	uint8_t *deviceSecond = copyToDevice(second);
	uint8_t *binary = NULL;
	uint8_t *difference = NULL;
	CUDA_CHECK(cudaMalloc(&binary, input.size()));
	CUDA_CHECK(cudaMalloc(&difference, input.size()));

	ImageProcessing processing;
	processing.blackwhiteGPU(deviceInput, binary, 128, width, height, false);
	processing.subtrachtImagesGPU(deviceSecond, deviceInput, difference, width, height, false);
	CUDA_CHECK(cudaThreadSynchronize());

	requirePixel(copyFromDevice(binary, input.size()),
		{0, 0, 255, 255, 255, 0}, "blackwhiteGPU");
	requirePixel(copyFromDevice(difference, input.size()),
		{10, 0, 0, 50, 0, 5}, "subtrachtImagesGPU");

	CUDA_CHECK(cudaFree(deviceInput));
	CUDA_CHECK(cudaFree(deviceSecond));
	CUDA_CHECK(cudaFree(binary));
	CUDA_CHECK(cudaFree(difference));
}

TEST(testLinearTransformation)
{
	const int width = 3;
	const int height = 2;
	std::vector<uint8_t> input = {0, 100, 200, 50, 128, 255};
	uint8_t *deviceInput = copyToDevice(input);
	uint8_t *deviceOutput = NULL;
	CUDA_CHECK(cudaMalloc(&deviceOutput, input.size()));

	ImageProcessing processing;
	processing.linearTransformationGPU(deviceInput, deviceOutput,
		width, height, 2.0f, -50, false);
	CUDA_CHECK(cudaThreadSynchronize());

	requirePixel(copyFromDevice(deviceOutput, input.size()),
		{0, 150, 255, 50, 206, 255}, "linearTransformationGPU");
	CUDA_CHECK(cudaFree(deviceInput));
	CUDA_CHECK(cudaFree(deviceOutput));
}

TEST(testHistogramAndHistogramEqualization)
{
	const int width = 2;
	const int height = 2;
	std::vector<uint8_t> input = {0, 64, 64, 255};
	unsigned int hostHistogram[256] = {};
	uint8_t *deviceInput = copyToDevice(input);
	uint8_t *deviceOutput = NULL;
	CUDA_CHECK(cudaMalloc(&deviceOutput, input.size()));

	ImageProcessing processing;
	processing.histogramGPU(deviceInput, hostHistogram, width, height, false);
	require(hostHistogram[0] == 1, "histogram black count");
	require(hostHistogram[64] == 2, "histogram mid count");
	require(hostHistogram[255] == 1, "histogram white count");

	processing.histogramEqualizationGPU(deviceInput, deviceOutput,
		hostHistogram, width, height, false);
	CUDA_CHECK(cudaThreadSynchronize());
	std::vector<uint8_t> equalized = copyFromDevice(deviceOutput, input.size());
	require(equalized.size() == input.size(), "histogramEqualizationGPU output size");

	CUDA_CHECK(cudaFree(deviceInput));
	CUDA_CHECK(cudaFree(deviceOutput));
}

TEST(testMaxMeanAreaAndTexture)
{
	const int width = 3;
	const int height = 2;
	std::vector<uint8_t> input = {0, 10, 20, 30, 40, 50};
	uint8_t *deviceInput = copyToDevice(input);
	ImageProcessing processing;

	int maximum = -1;
	int maximumX = -1;
	int maximumY = -1;
	processing.maximalElementGPU(deviceInput, maximum, maximumX, maximumY,
		width, height, false);
	require(maximum == 50, "maximalElementGPU value");
	require(maximumX == 2, "maximalElementGPU x");
	require(maximumY == 1, "maximalElementGPU y");

	int mean = -1;
	processing.meanVaueGPU(deviceInput, mean, width, height, false);
	require(mean == 25, "meanVaueGPU");

	double area = -1.0;
	std::vector<uint8_t> areaInput = {255, 255, 0, 255, 0, 0};
	uint8_t *deviceArea = copyToDevice(areaInput);
	processing.getAreaGPU(deviceArea, area, width, height, false);
	require(area == 3.0, "getAreaGPU");

	double texture = -1.0;
	processing.textureGPU(deviceInput, texture, width, height, false);
	require(std::isfinite(texture), "textureGPU finite result");

	CUDA_CHECK(cudaFree(deviceInput));
	CUDA_CHECK(cudaFree(deviceArea));
}

TEST(testGaussianConstantImage)
{
	const int width = 5;
	const int height = 5;
	std::vector<uint8_t> input(width * height, 255);
	uint8_t *deviceInput = copyToDevice(input);
	uint8_t *deviceOutput = NULL;
	CUDA_CHECK(cudaMalloc(&deviceOutput, input.size()));

	ImageProcessing processing;
	processing.gaussianBlurGPU(deviceInput, deviceOutput, width, height, 2, false);
	CUDA_CHECK(cudaThreadSynchronize());

	requirePixel(copyFromDevice(deviceOutput, input.size()), input,
		"gaussianBlurGPU constant image");
	CUDA_CHECK(cudaFree(deviceInput));
	CUDA_CHECK(cudaFree(deviceOutput));
}

int main()
{
	std::printf("ImageProcessing CUDA tests\n");

	int deviceCount = 0;
	cudaError_t deviceError = cudaGetDeviceCount(&deviceCount);
	if (deviceError != cudaSuccess || deviceCount == 0) {
		std::fprintf(stderr, "No CUDA device available: %s\n",
			cudaGetErrorString(deviceError));
		return EXIT_FAILURE;
	}

	RUN_TEST(testInvert);
	RUN_TEST(testBlackWhiteAndSubtract);
	RUN_TEST(testLinearTransformation);
	RUN_TEST(testHistogramAndHistogramEqualization);
	RUN_TEST(testMaxMeanAreaAndTexture);
	RUN_TEST(testGaussianConstantImage);

	if (failures != 0) {
		std::fprintf(stderr, "\n%d test assertion(s) failed\n", failures);
		return EXIT_FAILURE;
	}
	std::printf("All ImageProcessing CUDA tests passed\n");
	return EXIT_SUCCESS;
}
