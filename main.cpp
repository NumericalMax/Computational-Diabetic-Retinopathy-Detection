#include <iostream>
#include <string>
#include <ctime>

#include <sstream>
#include <fstream>

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
	cout << "Directory to labels: " << endl;
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


/*
Saves the featureVector as textfile to the specified destinaton
The format is hereby a tableformat seperated by comma, so that e.g. the data
can easily read by matlab or excel.

For program execution this wouldn't be necessary, but becomes advicable for manual computation.

@param destination Desired location, where the featureVector shall be stored as .txt

*/
void saveFeatureVector(string destination, string suffix, vector<string> name, string label, vector<double> featureVector){
	
	vector<string> example;
	ostringstream convert;

	example.push_back("");
	example.push_back("Image");
	example.push_back("Class");
	example.push_back("Bloodvessels");
	example.push_back("Haemorrhages");
	example.push_back("Exudates");
	example.push_back("Contrast");

	for (unsigned int i = 0; i < featureVector.size(); i++){

		if (i % 4 == 0){
			example.push_back("\n");
			example.push_back(name.at(i/4));
			example.push_back(label);

			convert.str("");
			convert.clear();
		}

		convert << featureVector.at(i);
		example.push_back(convert.str());

		convert.str("");
		convert.clear();
	}
        
        string fullPath = destination + "feature" + suffix + ".txt";
        
        char * writable = new char[fullPath.size() + 1];
        std::copy(fullPath.begin(), fullPath.end(), writable);
        writable[fullPath.size()] = '\0';
        
        ofstream output_file(writable, ios_base::out);
        
	ostream_iterator<string> output_iterator(output_file, ",");
	copy(example.begin(), example.end(), output_iterator);
}


int main(){
	
	vector<double> featureVector;
	ImageProcessing imgProcessor;
	uint8_t *host_imgdata, *device_orgdata, *device_imgdata, *device_imgdata1, *device_binaryMask;
	int bufSize, comps, width, height, length;
	double elapsedTime = 0.0;
	bool timing = true;	
	int max, mean, x, y;
	double bloodvessels, exudate, haemorrhage, contrast;
	unsigned int h_Result[256];

	cout << "===================================================================" << endl;
	cout << "Image Processing Library for Diabetic Retinopathy Detection" << endl;
	cout << "Version: 2.0" << endl;
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
    	cout << "Save feature matrix at: " << endl;
	cin >> directoryDest;
	cout << "===================================================================" << endl;    	
    
	readInLabels(directoryRes);

	length = images.size();

	vector<string> usedImages;
	clock_t beginLoad = clock();
	int i = 0;
	
	int upperLimit, lowerLimit;
	double upperScale = 3.0;
	double lowerScale = 0.7;
	double limit;

	while(i < length){
		
		cout << "-------------------------------------------------------------------" << endl;
		cout << i << "/" << length << " : " << images.at(i).c_str() << endl;
		
                bloodvessels = 0.0;
                exudate = 0.0;
                haemorrhage = 0.0;
                contrast = 0.0;

		host_imgdata = decompress_jpeg_image_from_file(images.at(i).c_str(), &width, &height, &comps, 1);
		bufSize = width * height * sizeof(uint8_t);
		uint8_t *finalImage = new uint8_t[width*height]();
	
		upperLimit = 255;
		lowerLimit = 0;
		limit = 0.0;

		cudaMalloc((void **)&device_orgdata, bufSize);
		cudaMalloc((void **)&device_imgdata, bufSize);
		cudaMalloc((void **)&device_imgdata1, bufSize);
		cudaMalloc((void **)&device_binaryMask, bufSize);	   

		cudaMemcpy(device_orgdata, host_imgdata, bufSize, cudaMemcpyHostToDevice);
		cudaMemcpy(device_imgdata, device_orgdata, bufSize, cudaMemcpyDeviceToDevice);
		cudaMemcpy(device_imgdata1, device_orgdata, bufSize, cudaMemcpyDeviceToDevice);
		cudaMemcpy(device_binaryMask, device_orgdata, bufSize, cudaMemcpyDeviceToDevice);
		cudaThreadSynchronize();
	
		// Remove comment for Time Measurement
		/*imgProcessor.histogramGPU(device_orgdata, h_Result, width, height, timing);
		imgProcessor.histogramEqualizationGPU(device_orgdata, device_imgdata, h_Result, width, height, timing);
                imgProcessor.invertGPU(device_orgdata, device_imgdata, width, height, timing);
		imgProcessor.dilateGPU(device_orgdata, device_imgdata, 5, width, height, timing);
                imgProcessor.erodeGPU(device_orgdata, device_imgdata, 5, width, height, timing);
		imgProcessor.gaussianBlurGPU(device_orgdata, device_imgdata1, width, height, 50, timing);
		imgProcessor.imadjustGPU(device_orgdata, device_imgdata, width, height, lowerLimit, upperLimit, lowerScale, upperScale, timing);
		imgProcessor.blackwhiteGPU(device_orgdata, device_imgdata, 125, width, height, timing);
		imgProcessor.maximalElementGPU(device_orgdata, max, x, y, width, height, timing);
        	imgProcessor.meanVaueGPU(device_orgdata, mean, width, height, timing);
		imgProcessor.textureGPU(device_orgdata, contrast, width, height, timing);
		imgProcessor.subtrachtImagesGPU(device_orgdata, device_imgdata1, device_imgdata, width, height, timing);
		imgProcessor.linearTransformationGPU(device_orgdata, device_imgdata, width, height, 2.0, 0, timing);
		*/
		
		//Feature Extraction PipeLine

		// Binary Mask
		imgProcessor.invertGPU(device_binaryMask, device_binaryMask, width, height, timing);
		imgProcessor.blackwhiteGPU(device_binaryMask, device_binaryMask, 250, width, height, timing);
		imgProcessor.dilateGPU(device_binaryMask, device_binaryMask, 20, width, height, timing);

		// Bloodvessels
		imgProcessor.invertGPU(device_orgdata, device_imgdata, width, height, timing);
		imgProcessor.dilateGPU(device_imgdata, device_imgdata1, 7, width, height, timing);
		imgProcessor.dilateGPU(device_imgdata, device_imgdata, 2, width, height, timing);
		imgProcessor.subtrachtImagesGPU(device_imgdata1, device_imgdata, device_imgdata, width, height, timing);
		imgProcessor.histogramGPU(device_imgdata, h_Result, width, height, timing);
                cudaDeviceSynchronize();
                while(limit < 0.15){
                        for(int k = 255; k >= upperLimit; k--){
                                limit += h_Result[k];
                        }

                        limit = limit / double((width * height));
                        upperLimit--;
                }
                limit = 0.0;
                while(limit < 0.15){
                        for(int k = 0; k <= lowerLimit; k++){
                                limit += h_Result[k];
                        }

                        limit = limit / double((width * height));
                        lowerLimit++;
                }
                imgProcessor.imadjustGPU(device_imgdata, device_imgdata, width, height, lowerLimit, 45, lowerScale, upperScale, timing);
		imgProcessor.linearTransformationGPU(device_orgdata, device_imgdata1, width, height, 0.1, 0, timing);
		imgProcessor.subtrachtImagesGPU(device_imgdata, device_imgdata1, device_imgdata, width, height, timing);
		imgProcessor.blackwhiteGPU(device_imgdata, device_imgdata, 10, width, height, timing);
		imgProcessor.getAreaGPU(device_imgdata, bloodvessels, width, height, timing);		

		// Exudate
		imgProcessor.gaussianBlurGPU(device_orgdata, device_imgdata, width, height, 24, timing);		
		imgProcessor.subtrachtImagesGPU(device_orgdata, device_imgdata, device_imgdata, width, height, timing);
		imgProcessor.histogramGPU(device_imgdata, h_Result, width, height, timing);
		cudaDeviceSynchronize();
                while(limit < 0.05){
                        for(int k = 255; k >= upperLimit; k--){
                                limit += h_Result[k];
                        }

                        limit = limit / double((width * height));
                        upperLimit--;
                }
                limit = 0.0;
                while(limit < 0.02){
                        for(int k = 0; k <= lowerLimit; k++){
                                limit += h_Result[k];
                        }

                        limit = limit / double((width * height));
                        lowerLimit++;
                }
                imgProcessor.imadjustGPU(device_imgdata, device_imgdata, width, height, lowerLimit, upperLimit, lowerScale, upperScale, timing);
		imgProcessor.dilateGPU(device_imgdata, device_imgdata, 14, width, height, timing);
		imgProcessor.gaussianBlurGPU(device_orgdata, device_imgdata1, width, height, 24, timing);
                imgProcessor.subtrachtImagesGPU(device_imgdata, device_imgdata1, device_imgdata, width, height, timing);
		imgProcessor.linearTransformationGPU(device_imgdata, device_imgdata, width, height, 2.0, 0, timing);
		imgProcessor.blackwhiteGPU(device_imgdata, device_imgdata, 80, width, height, timing);
		imgProcessor.getAreaGPU(device_imgdata, exudate, width, height, timing);

		// Haemorrhages
		imgProcessor.invertGPU(device_orgdata, device_imgdata, width, height, timing);
		imgProcessor.gaussianBlurGPU(device_orgdata, device_imgdata1, width, height, 24, timing);
		imgProcessor.subtrachtImagesGPU(device_imgdata, device_imgdata1, device_imgdata, width, height, timing);
		imgProcessor.histogramGPU(device_imgdata, h_Result, width, height, timing);
                cudaDeviceSynchronize();
                while(limit < 0.10){
                        for(int k = 255; k >= upperLimit; k--){
                                limit += h_Result[k];
                        }

                        limit = limit / double((width * height));
                        upperLimit--;
                }
                limit = 0.0;
                while(limit < 0.25){
                        for(int k = 0; k <= lowerLimit; k++){
                                limit += h_Result[k];
                        }

                        limit = limit / double((width * height));
                        lowerLimit++;
                }


		cout << "Lower Limit: " << lowerLimit << " Upper Limit: " << upperLimit << endl;

                imgProcessor.imadjustGPU(device_imgdata, device_imgdata, width, height, lowerLimit, 40, lowerScale, upperScale, timing);
		imgProcessor.erodeGPU(device_imgdata, device_imgdata, 12, width, height, timing);
		imgProcessor.dilateGPU(device_imgdata, device_imgdata, 6, width, height, timing);
		imgProcessor.linearTransformationGPU(device_imgdata, device_imgdata, width, height, 2.0, 0, timing);
		imgProcessor.blackwhiteGPU(device_imgdata, device_imgdata, 20, width, height, timing);
		imgProcessor.subtrachtImagesGPU(device_imgdata, device_binaryMask, device_imgdata, width, height, timing);
                imgProcessor.getAreaGPU(device_imgdata, haemorrhage, width, height, timing);

		//Contrast
		imgProcessor.textureGPU(device_orgdata, contrast, width, height, timing);
		
		cudaThreadSynchronize();

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess){
			printf("Error: %s\n", cudaGetErrorString(err));
		}

		cudaThreadSynchronize();

		delete finalImage;
		delete host_imgdata;

		cudaFree(device_orgdata);
		cudaFree(device_imgdata);
		cudaFree(device_imgdata1);
		cudaFree(device_binaryMask);

		bloodvessels = bloodvessels/(width*height);
		exudate = exudate / (width*height);
		haemorrhage = haemorrhage / (width*height);
	
		cudaThreadSynchronize();
		usedImages.push_back(images.at(i));
		featureVector.push_back(bloodvessels);
		featureVector.push_back(haemorrhage);
		featureVector.push_back(exudate);
		featureVector.push_back(contrast);
		i++;
	
		cout << "-------------------------------------------------------------------" << endl;
	
	}

	saveFeatureVector(directoryDest, suff, usedImages, suff, featureVector);
	clock_t endLoad = clock();
	elapsedTime = double(endLoad - beginLoad) / CLOCKS_PER_SEC;

	cout << "===================================================================" << endl << endl;
	cout << "Elapsed time: " << elapsedTime << " seconds." << endl;
	cout << "Finished successfully!" << endl << endl;
	cout << "===================================================================" << endl << endl;

	return 0;
}
