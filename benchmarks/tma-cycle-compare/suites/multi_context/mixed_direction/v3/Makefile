VENTUS_ROOT := $(abspath ../../../../../..)
CXX ?= clang++
CXXFLAGS := -O2 -g -std=c++17 -DCL_TARGET_OPENCL_VERSION=120 \
	-Wno-deprecated-declarations
TARGET := ventus_multi_context
COMMON := $(VENTUS_ROOT)/testcases/_get_case/common
NVCC ?= /usr/local/cuda-13.1/bin/nvcc

.PHONY: all validate clean

all: $(TARGET)

$(TARGET): ventus_multi_context.cpp ventus_multi_context.cl
	$(CXX) $(CXXFLAGS) ventus_multi_context.cpp -o $@ \
		-I$(COMMON) -I$(VENTUS_ROOT)/install/include \
		-L$(VENTUS_ROOT)/install/lib \
		-Wl,-rpath,$(VENTUS_ROOT)/install/lib -lOpenCL

validate: $(TARGET)
	$(VENTUS_ROOT)/install/bin/clang -x cl -cl-std=CL2.0 -target riscv32 \
		-mcpu=ventus-gpgpu -D__cl_clang_storage_class_specifiers \
		-D__opencl_c_generic_address_space=1 \
		-D__opencl_c_named_address_space_builtins=1 \
		-D__OPENCL_VERSION__=200 -I$(COMMON) -I. -fsyntax-only \
		ventus_multi_context.cl
	$(NVCC) -O3 -std=c++20 -arch=sm_90 cuda_multi_context.cu \
		-o /tmp/tma_multi_v2_sm90 -lcuda
	$(NVCC) -O3 -std=c++20 -arch=sm_100 cuda_multi_context.cu \
		-o /tmp/tma_multi_v2_sm100 -lcuda
	python3 -m py_compile modal_multi_context.py run_ventus_multi_context_short.py

clean:
	rm -f $(TARGET)
