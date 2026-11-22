NVCC=nvcc
ARCH=sm_35

all:
	$(NVCC) -O3 -x=cu -arch=$(ARCH) main.cpp ImageProcessing.cpp Jpgd.cpp Jpge.cpp -o preProcessing

test:
	$(NVCC) -O3 -std=c++11 -x=cu -arch=$(ARCH) tests/ImageProcessingTest.cpp ImageProcessing.cpp -o imageProcessingTest
	./imageProcessingTest

clean:
	rm -f preProcessing imageProcessingTest
