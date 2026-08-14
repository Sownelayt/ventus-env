NVCC ?= /usr/local/cuda-13.1/bin/nvcc
SOURCE := cuda_control_items.cu
OUT := /tmp/tma-control-items-v1

.PHONY: validate sm90 sm100 python

validate: sm90 sm100 python

sm90:
	$(NVCC) -O3 -lineinfo -std=c++20 -arch=sm_90 $(SOURCE) -o $(OUT)-sm90 -lcuda
	$(NVCC) -O3 -lineinfo -std=c++20 -arch=sm_90 $(SOURCE) -ptx -o $(OUT)-sm90.ptx -lcuda
	grep -q "cp.async.bulk.tensor" $(OUT)-sm90.ptx
	grep -q "cp.async.bulk.wait_group" $(OUT)-sm90.ptx

sm100:
	$(NVCC) -O3 -lineinfo -std=c++20 -arch=sm_100 $(SOURCE) -o $(OUT)-sm100 -lcuda
	$(NVCC) -O3 -lineinfo -std=c++20 -arch=sm_100 $(SOURCE) -ptx -o $(OUT)-sm100.ptx -lcuda
	grep -q "cp.async.bulk.tensor" $(OUT)-sm100.ptx
	grep -q "cp.async.bulk.wait_group" $(OUT)-sm100.ptx

python:
	python3 -m py_compile modal_control_items.py
