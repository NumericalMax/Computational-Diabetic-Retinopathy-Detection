#include <iostream>
#include <string>
#include <ctime>

#include <sstream>
#include <fstream>
#include <cstdlib>

#include "ImageProcessing.h"
#include "Jpgd.h"
#include "Jpge.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_device_runtime_api.h>

using namespace std;
using namespace jpgd;
using namespace jpge;

vector<string> images;
vector<string> labels;

/*
This functions checks, whether all CUDA - requirements are fullfilled.
Further it prints out informations about the current CUDA settings.

@returns bool whether all CUDA - requirements are fullfilled.
*/
bool cudaConfiguration(){

	int deviceCount = 0;

	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess){
		cout << "On this system is no CUDA device available." << endl;
		cin.get();
		return false;
	}

	if (deviceCount == 0){
		cout << "On this system is no CUDA device available." << endl;
		cin.get();
		return false;
	}
	else if (deviceCount == 1){
		cout << "===================================================================" << endl;
		cout << "On this system is " << deviceCount << " CUDA device available." << endl;
		cout << "===================================================================" << endl;
	}
	else if (deviceCount > 1){
		cout << "===================================================================" << endl;
		cout << "On this system are " << deviceCount << " CUDA devices available." << endl;
		cout << "===================================================================" << endl;
	}

	int device = 0;
	cudaSetDevice(device);

	cudaDeviceProp p;
	cudaGetDeviceProperties(&p, device);
	
	cout << "===================================================================" << endl;
	cout << "Device: " << p.name << "  ";
	cout << "Streaming MP: " << p.multiProcessorCount << "  ";
	cout << "Compute Version: " << p.major << "." << p.minor << endl;
	cout << "===================================================================" << endl;

	return true;
}

/*
 * Based on the input direction, labels inside the a .txt file are loaded into memory.
 * Afterwards the the label strings and the folder directory are put together in Order
 * to gain the complete path to each image.
 *
 * @param void
 * */
void readInLabels(string resDirectory){

	string line;
    string directory;
	cout << "Directory to labels .txt File: " << endl;
	cin >> directory;
	cout << endl;

	ifstream fe(directory.c_str(), std::ios_base::in);
    	ifstream f;

	if (fe.is_open()){
		while (getline(fe, line, ';')){
			string suffix (".jpeg");
			string dir(resDirectory);
			string path(dir + line + suffix);
			
			f.open(path.c_str());

			if (f.good()) {
				
				images.push_back(path);
				labels.push_back(line);
	
			}

			f.close();

		}
		fe.close();
	}
	
	else{
		cout << "Unable to open file" << endl;	
	}
}


int main(){
	
	ImageProcessing imgProcessor;
	uint8_t *host_imgdata = NULL, *device_orgdata = NULL, *device_imgdata = NULL, *device_imgdata1 = NULL, *device_imgdata2 = NULL;
	int bufSize, comps, width, height, length;
	bool timing = true;	

	cout << "===================================================================" << endl;
	cout << "Image Processing Library" << endl;
	cout << "Version: 1.0" << endl;
	cout << "===================================================================" << endl;

    	if(!cudaConfiguration()){return 0;};

	string suff;

	cout << "===================================================================" << endl;
	cout << "Enter Diabetic Retinopathy Stage: ";
	cin >> suff;
	cout << "===================================================================" << endl;	
	
	cout << "===================================================================" << endl;
    string directoryRes;
	cout << "Directory to image resource: " << endl;
	cin >> directoryRes;
	cout << "===================================================================" << endl;
    	
	cout << "===================================================================" << endl;
    string directoryDest;
    cout << "Save processed images at: " << endl;
	cin >> directoryDest;
	cout << "===================================================================" << endl;    	
	readInLabels(directoryRes);

	length = images.size();

	vector<string> usedImages;
	clock_t beginLoad = clock();
	int i = 0;
	const int req_comps = 3;
	while(i < length){

		cout << "-------------------------------------------------------------------" << endl;
		cout << i << "/" << length << " : " << images.at(i).c_str() << endl;
		
		host_imgdata = decompress_jpeg_image_from_file(images.at(i).c_str(), &width, &height, &comps, 1);
		if (host_imgdata == NULL){
			cerr << "Unable to decode JPEG: " << images.at(i) << endl;
			i++;
			continue;
		}
		if (width <= 0 || height <= 0){
			cerr << "Invalid JPEG dimensions: " << images.at(i) << endl;
			free(host_imgdata);
			i++;
			continue;
		}

		cout << width *height * comps << endl;

		bufSize = width * height * sizeof(uint8_t);
		uint8_t *finalImage = new uint8_t[width*height*3]();
		uint8_t *finalImageEx = new uint8_t[width*height]();
		uint8_t *finalImageH = new uint8_t[width*height]();
		uint8_t *finalImageM = new uint8_t[width*height]();	

		if (cudaMalloc((void **)&device_orgdata, bufSize) != cudaSuccess ||
		    cudaMalloc((void **)&device_imgdata, bufSize) != cudaSuccess ||
		    cudaMalloc((void **)&device_imgdata1, bufSize) != cudaSuccess ||
		    cudaMalloc((void **)&device_imgdata2, bufSize) != cudaSuccess){
			cerr << "Unable to allocate CUDA image buffers" << endl;
			cudaFree(device_orgdata);
			cudaFree(device_imgdata);
			cudaFree(device_imgdata1);
			cudaFree(device_imgdata2);
			free(host_imgdata);
			i++;
			continue;
		}

		if (cudaMemcpy(device_orgdata, host_imgdata, bufSize, cudaMemcpyHostToDevice) != cudaSuccess ||
		    cudaMemcpy(device_imgdata, device_orgdata, bufSize, cudaMemcpyDeviceToDevice) != cudaSuccess ||
		    cudaMemcpy(device_imgdata1, device_orgdata, bufSize, cudaMemcpyDeviceToDevice) != cudaSuccess ||
		    cudaMemcpy(device_imgdata2, device_orgdata, bufSize, cudaMemcpyDeviceToDevice) != cudaSuccess){
			cerr << "Unable to copy image data to CUDA" << endl;
			cudaFree(device_orgdata);
			cudaFree(device_imgdata);
			cudaFree(device_imgdata1);
			cudaFree(device_imgdata2);
			delete[] finalImage;
			delete[] finalImageEx;
			delete[] finalImageH;
			delete[] finalImageM;
			free(host_imgdata);
			i++;
			continue;
		}
		cudaThreadSynchronize();
		
		imgProcessor.invertGPU(device_orgdata, device_imgdata1, width, height, timing);
		imgProcessor.gaussianBlurGPU(device_imgdata1, device_imgdata2, width, height, 20, timing);
		imgProcessor.subtrachtImagesGPU(device_imgdata1, device_imgdata2, device_imgdata1, width, height, timing);

		imgProcessor.invertGPU(device_orgdata, device_imgdata2, width, height, timing);
		imgProcessor.gaussianBlurGPU(device_imgdata2, device_imgdata, width, height, 20, timing);
		imgProcessor.subtrachtImagesGPU(device_imgdata2, device_imgdata, device_imgdata2, width, height, timing);

		imgProcessor.gaussianBlurGPU(device_orgdata, device_imgdata, width, height, 20, timing);
		imgProcessor.subtrachtImagesGPU(device_orgdata, device_imgdata, device_imgdata, width, height, timing);

		
		imgProcessor.erodeGPU(device_imgdata, device_imgdata, 3, width, height, timing);
		imgProcessor.erodeGPU(device_imgdata1, device_imgdata1, 2, width, height, timing);
		imgProcessor.erodeGPU(device_imgdata2, device_imgdata2, 2, width, height, timing);
		
		imgProcessor.linearTransformationGPU(device_imgdata, device_imgdata, width, height, 15, 0, timing);
		imgProcessor.linearTransformationGPU(device_imgdata1, device_imgdata1, width, height, 15, 0, timing);
		imgProcessor.linearTransformationGPU(device_imgdata2, device_imgdata2, width, height, 15, 0, timing);

		cudaThreadSynchronize();

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess){
			printf("Error: %s\n", cudaGetErrorString(err));
		}

		cudaThreadSynchronize();

		if (cudaMemcpy(finalImageEx, device_imgdata, width*height*sizeof(uint8_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
		    cudaMemcpy(finalImageH, device_imgdata1, width*height*sizeof(uint8_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
		    cudaMemcpy(finalImageM, device_imgdata2, width*height*sizeof(uint8_t), cudaMemcpyDeviceToHost) != cudaSuccess){
			cerr << "Unable to copy processed image data to host" << endl;
			delete[] finalImageEx;
			delete[] finalImageH;
			delete[] finalImageM;
			delete[] finalImage;
			free(host_imgdata);
			cudaFree(device_orgdata);
			cudaFree(device_imgdata);
			cudaFree(device_imgdata1);
			cudaFree(device_imgdata2);
			i++;
			continue;
		}

		for(int l = 0; l < height*width; l ++){
				
			finalImage[3*l] = finalImageH[l];
			finalImage[3*l + 1] = finalImageEx[l];
			finalImage[3*l + 2] = finalImageM[l];

		}
		
		string saveAt(directoryDest + labels.at(i) + ".jpeg");
		compress_image_to_jpeg_file(saveAt.c_str(), width, height, 3, finalImage);
		
		delete[] finalImageEx;
		delete[] finalImageH;
		delete[] finalImageM;
		delete[] finalImage;
		free(host_imgdata);

		cudaFree(device_orgdata);
		cudaFree(device_imgdata);
		cudaFree(device_imgdata1);
        cudaFree(device_imgdata2);
	
		cudaThreadSynchronize();
		i++;
	
		cout << "-------------------------------------------------------------------" << endl;
	
	}

	cout << "===================================================================" << endl << endl;
	cout << "Finished successfully!" << endl << endl;
	cout << "===================================================================" << endl << endl;

	return 0;
}
