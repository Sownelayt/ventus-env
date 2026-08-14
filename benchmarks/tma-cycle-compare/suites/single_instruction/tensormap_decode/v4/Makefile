VENTUS_ROOT := $(abspath ../../../../../..)
CXX ?= clang++
CXXFLAGS := -O2 -g -std=c++17 -DCL_TARGET_OPENCL_VERSION=120 \
	-Wno-deprecated-declarations
TARGET := ventus_tensormap_decode
COMMON := $(VENTUS_ROOT)/testcases/_get_case/common
NVCC ?= /usr/local/cuda-13.1/bin/nvcc

.PHONY: all validate clean

all: $(TARGET)

$(TARGET): ventus_tensormap_decode.cpp ventus_tensormap_decode.cl
	$(CXX) $(CXXFLAGS) ventus_tensormap_decode.cpp -o $@ \
		-I$(COMMON) -I$(VENTUS_ROOT)/install/include \
		-L$(VENTUS_ROOT)/install/lib \
		-Wl,-rpath,$(VENTUS_ROOT)/install/lib -lOpenCL

validate: $(TARGET)
	$(VENTUS_ROOT)/install/bin/clang -x cl -cl-std=CL2.0 -target riscv32 \
		-mcpu=ventus-gpgpu -D__cl_clang_storage_class_specifiers \
		-D__opencl_c_generic_address_space=1 \
		-D__opencl_c_named_address_space_builtins=1 \
		-D__OPENCL_VERSION__=200 -I$(COMMON) -I. -fsyntax-only \
		ventus_tensormap_decode.cl
	$(NVCC) -O3 -std=c++20 -arch=sm_90 cuda_tensormap_decode.cu \
		-o /tmp/tma_decode_v3_sm90 -lcuda
	$(NVCC) -O3 -std=c++20 -arch=sm_90 cuda_tensormap_decode.cu \
		-cubin -o /tmp/tma_decode_v3_sm90.cubin -lcuda
	/usr/local/cuda-13.1/bin/cuobjdump --dump-sass \
		/tmp/tma_decode_v3_sm90.cubin > /tmp/tma_decode_v3_sm90.sass
	python3 audit_instruction_stream.py cuda_tensormap_decode.cu \
		/tmp/tma_decode_v3_sm90.sass --output /tmp/tma_decode_v3_sm90.audit.json
	/usr/local/cuda-13.1/bin/nvdisasm /tmp/tma_decode_v3_sm90.cubin \
		> /tmp/tma_decode_v3_sm90.nvdisasm.sass
	python3 audit_instruction_stream.py cuda_tensormap_decode.cu \
		/tmp/tma_decode_v3_sm90.nvdisasm.sass \
		--output /tmp/tma_decode_v3_sm90.nvdisasm.audit.json
	$(NVCC) -O3 -std=c++20 -arch=sm_100 cuda_tensormap_decode.cu \
		-o /tmp/tma_decode_v3_sm100 -lcuda
	python3 -m py_compile modal_tensormap_decode.py run_ventus_decode_short.py \
		audit_instruction_stream.py

clean:
	rm -f $(TARGET)
