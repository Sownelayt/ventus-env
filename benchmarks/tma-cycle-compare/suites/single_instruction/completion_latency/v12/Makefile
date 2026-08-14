NVCC ?= /usr/local/cuda-13.1/bin/nvcc
CUOBJDUMP ?= /usr/local/cuda-13.1/bin/cuobjdump
ARCH ?= sm_90
ROOT := $(abspath ../../../../../..)
TMA_REFER ?= /home/liyb/tma-refer
TARGET := cuda_g2s_fresh_map
INCLUDES := -I$(TMA_REFER)/src -I../../common_matrix/v3 -I$(ROOT)/testcases/_get_case/common

.PHONY: all list audit clean

all: $(TARGET) $(TARGET).ptx $(TARGET).cubin $(TARGET).sass audit

$(TARGET): cuda_g2s_fresh_map.cu delay_ladder_v12.inc
	$(NVCC) -O3 -std=c++20 -arch=$(ARCH) $(INCLUDES) $< -o $@ -L/usr/local/cuda-13.1/lib64/stubs -lcuda

$(TARGET).ptx: cuda_g2s_fresh_map.cu delay_ladder_v12.inc
	$(NVCC) -O3 -std=c++20 -arch=$(ARCH) $(INCLUDES) $< -ptx -o $@ -lcuda

$(TARGET).cubin: cuda_g2s_fresh_map.cu delay_ladder_v12.inc
	$(NVCC) -O3 -std=c++20 -arch=$(ARCH) $(INCLUDES) $< -cubin -o $@ -lcuda

$(TARGET).sass: $(TARGET).cubin
	$(CUOBJDUMP) --dump-sass $< > $@

list: $(TARGET)
	./$(TARGET) --project common --list
	./$(TARGET) --project tensor_2d --list

audit: $(TARGET).ptx $(TARGET).sass
	python3 audit_instruction_stream.py $(TARGET).ptx $(TARGET).sass --output instruction_audit.json

clean:
	rm -f $(TARGET) $(TARGET).ptx $(TARGET).cubin $(TARGET).sass instruction_audit.json
