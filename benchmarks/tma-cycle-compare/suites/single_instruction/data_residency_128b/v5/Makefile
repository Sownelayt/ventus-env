NVCC ?= /usr/local/cuda-13.1/bin/nvcc

.PHONY: all validate

all: validate

validate:
	$(NVCC) -O3 -std=c++20 -arch=sm_90 cuda_data_residency_128b.cu \
		-o /tmp/tma_residency_v5_sm90 -lcuda
	$(NVCC) -O3 -std=c++20 -arch=sm_100 cuda_data_residency_128b.cu \
		-o /tmp/tma_residency_v5_sm100 -lcuda
	python3 -m py_compile modal_data_residency_128b.py
