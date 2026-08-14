SUITE_ROOT := suites

.PHONY: list validate common-check tensor-2d-check decode-check residency-check \
	control-check completion-check multi-context-check

list:
	python3 run.py --list

validate: common-check tensor-2d-check decode-check residency-check control-check completion-check multi-context-check

common-check:
	$(MAKE) -C $(SUITE_ROOT)/single_instruction/common_matrix/v4 validate

tensor-2d-check:
	$(MAKE) -C $(SUITE_ROOT)/single_instruction/tensor_2d_capacity/v4 validate

decode-check:
	$(MAKE) -C $(SUITE_ROOT)/single_instruction/tensormap_decode/v4 validate

residency-check:
	$(MAKE) -C $(SUITE_ROOT)/single_instruction/data_residency_128b/v5 validate

control-check:
	$(MAKE) -C $(SUITE_ROOT)/single_instruction/control_items/v3 validate

completion-check:
	$(MAKE) -C $(SUITE_ROOT)/single_instruction/completion_latency/v12 all

multi-context-check:
	$(MAKE) -C $(SUITE_ROOT)/multi_context/same_direction/v4 validate
	$(MAKE) -C $(SUITE_ROOT)/multi_context/mixed_direction/v3 validate
