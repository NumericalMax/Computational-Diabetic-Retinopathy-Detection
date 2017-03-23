#ReadMe: A Feture Extraction Approach for the Computational Diabetic Retinopathy Detection in Retinal Images

Subsequent an Instruction is provided, how results within the paper can be reproduced.
Remark, that GPU and CPU related proceedings were accomplished on a linux based machine.

Software Requirements

	•	Matlab
	•	CUDA Toolkit (Version >= 7.0)

Hardware Requirements

	•	NVIDIA GPU with CUDA
        	Minimum Architecture: 2.0 (not tested / makefile has to be adapted)
        	Suggested Architecture: >= 3.5
	•	Modern CPU

Data Resource

	•	Jorge Cuadros and George Bresnick Eyepacs: An adaptable telemedicine system for diabetic retinopathy screening Journal of diabetes science and technology (Online), 3(3):509–516, 05 200
	•	Further information: https://www.kaggle.com/c/diabetic-retinopathy-detection

Data Format and Structure

	•	88702 JPEG Images
	•	Feature Extraction on GPU
		⁃	Folder with Subfolders
		⁃	Each Folder represents a particular Stage of DR (e.g. ../RetinalImages/0/ ../RetinalImages/1/ ../RetinalImages/2/ ../RetinalImages/3/ ../RetinalImages/4/)
		⁃	Retinal Grey Scale Images (Suggested: Green Component) are within the corresponding Folder 
		⁃	.txt File with labels holding following Format:
			10_left,10_right,11_left,11_right,…,30000_left
		⁃	A prior matching evaluates whether a particular label is contained within the folder, hence number of labels do not have to match number of images in the referrenced folder
	•	Feature Extraction on CPU
		⁃	Folder with Subfolders
		⁃	Each Folder represents a particular Stage of DR (e.g. ../RetinalImages/0/ ../RetinalImages/1/ ../RetinalImages/2/ ../RetinalImages/3/ ../RetinalImages/4/)
		⁃	Retinal Grey Scale Images (Suggested: Green Component) are within the corresponding Folder 

SVM Approach / Feature Extraction

	•	Matlab / CPU
		⁃	Primary used for comfortable Visualization of Results and Establishment of Image Processing Pipeline
		⁃	Open File ../CPU/featureExtraction.m in Matlab
		⁃	Execute
				function [ MAT, rowValue ] = featureExtraction(imagePath, class, destinationPath, plot)

		⁃	Corresponding Explanation of Variables is stated within the m-File

	•	CUDA / GPU
		⁃	Compile Software in Folder ./GPU/ with enclosed makefile (nvcc compiler required)
		⁃	Execute: ./featureExtraction
		⁃	Exemplary Run:
			===================================================================
			Image Processing Library
			Version: 1.0
		
			On this system are 4 CUDA devices available.
			Device: Tesla K20m  Streaming MP: 13  Compute Version: 3.5
		
			Directory to image resource: /home/kapsecker/imagesGrey/3/
			Directory to image destination: /home/kapsecker/images/3/
			Directory to labels: /home/kapsecker/res/labels.txt
			-------------------------------------------------------------------
			0/2086 : /home/kapsecker/imagesGrey/3/99_left.jpeg
			1/2086 : /home/kapsecker/imagesGrey/3/99_right.jpeg
			2/2086 : /home/kapsecker/imagesGrey/3/163_left.jpeg
			…
			2085/2086 : /home/kapsecker/imagesGrey/3/44333_left.jpeg
			2086/2086 : /home/kapsecker/imagesGrey/3/44333_right.jpeg
			-------------------------------------------------------------------

			Elapsed time: 135.79 seconds.
			Finished successfully!
			===================================================================

SVM Approach / Classification:

	•	Resulting Feature Matrix is saved in following Form

			Image,Class,Bloodvessels,Haemorrhages,Exudates,Contrast,
			/home/kapsecker/imagesGrey/2/30_right.jpeg,2,0.000616252,0,0,0.255243,
			/home/kapsecker/imagesGrey/2/40_left.jpeg,2,4.10808e-05,0,0,0.273186,
			/home/kapsecker/imagesGrey/2/51_left.jpeg,2,4.98195e-06,0,0,0.116361,
			/home/kapsecker/imagesGrey/2/54_left.jpeg,2,0.00131854,0,0,0.175058,
			/home/kapsecker/imagesGrey/2/78_left.jpeg,2,0.000438692,0,0,0.114433,
			/home/kapsecker/imagesGrey/2/78_right.jpeg,2,0.000346607,0,0,0.109931,
			/home/kapsecker/imagesGrey/2/79_left.jpeg,2,4.25127e-06,0,0,0.191255,
			/home/kapsecker/imagesGrey/2/79_right.jpeg,2,8.10398e-06,0,0,0.270105,
			/home/kapsecker/imagesGrey/2/82_left.jpeg,2,0.00180359,0,0,0.650781,
			/home/kapsecker/imagesGrey/2/129_left.jpeg,2,8.80809e-05,0,0,0.135238,
			/home/kapsecker/imagesGrey/2/129_right.jpeg,2,0.000138166,0,0,0.148248,
			…

	•	Import resulting Feature Matrix to Matlab’s Classification Learner
	•	Train with desired Classification Learner

Results:

	•	Feature Matrices, obtained from Feature Extraction Process, are saved in ../Features/
		Remark, that they are distinguished by Diabetic Retinopathy Classes

Edit / Redistribution:

	•	Free for non commercial use
	•	Feel free to use, edit or redistribute the established ImageProcessing Library for scientific purpose
	•	Please refer to this repository.
