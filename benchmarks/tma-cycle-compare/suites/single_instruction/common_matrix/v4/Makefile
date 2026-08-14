VENTUS_ROOT := $(abspath ../../../../../..)
CXX ?= clang++
CXXFLAGS := -O2 -g -std=c++17 -DCL_TARGET_OPENCL_VERSION=120 \
	-Wno-deprecated-declarations
TARGET := ventus_tma_comprehensive
COMMON := $(VENTUS_ROOT)/testcases/_get_case/common
NVCC ?= /usr/local/cuda-13.1/bin/nvcc

.PHONY: all validate clean

all: $(TARGET)

$(TARGET): ventus_tma_comprehensive.cpp ventus_tma_comprehensive.cl \
		ventus_tma_feature_parity.cl tma_comprehensive_matrix.h \
		$(COMMON)/tma_model.cc
	$(CXX) $(CXXFLAGS) ventus_tma_comprehensive.cpp $(COMMON)/tma_model.cc \
		-o $@ -I$(COMMON) -I$(VENTUS_ROOT)/install/include \
		-L$(VENTUS_ROOT)/install/lib \
		-Wl,-rpath,$(VENTUS_ROOT)/install/lib -lOpenCL

validate: $(TARGET)
	@for shard in bulk tensor_capacity tensor_geometry dtype layout interleave16; do \
		./$(TARGET) --validate --shard $$shard || exit $$?; \
	done
	test "$$(for shard in bulk tensor_capacity tensor_geometry dtype layout interleave16; do \
		./$(TARGET) --list --shard $$shard; done | wc -l)" -eq 334
	$(VENTUS_ROOT)/install/bin/clang -x cl -cl-std=CL2.0 -target riscv32 \
		-mcpu=ventus-gpgpu -D__cl_clang_storage_class_specifiers \
		-D__opencl_c_generic_address_space=1 \
		-D__opencl_c_named_address_space_builtins=1 \
		-D__OPENCL_VERSION__=200 -I$(COMMON) -I. -fsyntax-only \
		ventus_tma_comprehensive.cl
	$(NVCC) -O2 -std=c++20 -arch=sm_90 -I/home/liyb/tma-refer/src \
		-I$(COMMON) cuda_tma_comprehensive.cu -o /tmp/tma_common_v3_sm90 -lcuda
	$(NVCC) -O2 -std=c++20 -arch=sm_90 -I/home/liyb/tma-refer/src \
		-I$(COMMON) cuda_tma_comprehensive.cu -ptx -o /tmp/tma_common_v3_sm90.ptx -lcuda
	$(NVCC) -O2 -std=c++20 -arch=sm_90 -I/home/liyb/tma-refer/src \
		-I$(COMMON) cuda_tma_comprehensive.cu -cubin -o /tmp/tma_common_v3_sm90.cubin -lcuda
	/usr/local/cuda-13.1/bin/cuobjdump --dump-sass /tmp/tma_common_v3_sm90.cubin \
		> /tmp/tma_common_v3_sm90.sass
	python3 audit_fresh_tensormap.py cuda_tma_comprehensive.cu \
		/tmp/tma_common_v3_sm90.ptx /tmp/tma_common_v3_sm90.sass \
		--output /tmp/tma_common_v3_sm90.audit.json
	/usr/local/cuda-13.1/bin/nvdisasm /tmp/tma_common_v3_sm90.cubin \
		> /tmp/tma_common_v3_sm90.nvdisasm.sass
	python3 audit_fresh_tensormap.py cuda_tma_comprehensive.cu \
		/tmp/tma_common_v3_sm90.ptx /tmp/tma_common_v3_sm90.nvdisasm.sass \
		--output /tmp/tma_common_v3_sm90.nvdisasm.audit.json
	$(NVCC) -O2 -std=c++20 -arch=sm_100 -I/home/liyb/tma-refer/src \
		-I$(COMMON) cuda_tma_comprehensive.cu -o /tmp/tma_common_v3_sm100 -lcuda
	python3 -m py_compile modal_tma_comprehensive_compare.py \
		run_ventus_comprehensive_short.py audit_fresh_tensormap.py

clean:
	rm -f $(TARGET)
