# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /export/users/zhaojiam/pkg/cmake-3.29.0-rc1-linux-x86_64/bin/cmake

# The command to remove a file.
RM = /export/users/zhaojiam/pkg/cmake-3.29.0-rc1-linux-x86_64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /export/users/zhaojiam/src/ByteTransformer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /export/users/zhaojiam/src/ByteTransformer/dpct

# Include any dependencies generated for this target.
include bytetransformer/src/CMakeFiles/bytetransformer.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include bytetransformer/src/CMakeFiles/bytetransformer.dir/compiler_depend.make

# Include the progress variables for this target.
include bytetransformer/src/CMakeFiles/bytetransformer.dir/progress.make

# Include the compile flags for this target's objects.
include bytetransformer/src/CMakeFiles/bytetransformer.dir/flags.make

bytetransformer/src/CMakeFiles/bytetransformer.dir/gemm.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/flags.make
bytetransformer/src/CMakeFiles/bytetransformer.dir/gemm.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/includes_CUDA.rsp
bytetransformer/src/CMakeFiles/bytetransformer.dir/gemm.cu.o: /export/users/zhaojiam/src/ByteTransformer/bytetransformer/src/gemm.cu
bytetransformer/src/CMakeFiles/bytetransformer.dir/gemm.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/export/users/zhaojiam/src/ByteTransformer/dpct/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object bytetransformer/src/CMakeFiles/bytetransformer.dir/gemm.cu.o"
	cd /export/users/zhaojiam/src/ByteTransformer/dpct/bytetransformer/src && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT bytetransformer/src/CMakeFiles/bytetransformer.dir/gemm.cu.o -MF CMakeFiles/bytetransformer.dir/gemm.cu.o.d -x cu -c /export/users/zhaojiam/src/ByteTransformer/bytetransformer/src/gemm.cu -o CMakeFiles/bytetransformer.dir/gemm.cu.o

bytetransformer/src/CMakeFiles/bytetransformer.dir/gemm.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/bytetransformer.dir/gemm.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

bytetransformer/src/CMakeFiles/bytetransformer.dir/gemm.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/bytetransformer.dir/gemm.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

bytetransformer/src/CMakeFiles/bytetransformer.dir/gemm_bias_act.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/flags.make
bytetransformer/src/CMakeFiles/bytetransformer.dir/gemm_bias_act.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/includes_CUDA.rsp
bytetransformer/src/CMakeFiles/bytetransformer.dir/gemm_bias_act.cu.o: /export/users/zhaojiam/src/ByteTransformer/bytetransformer/src/gemm_bias_act.cu
bytetransformer/src/CMakeFiles/bytetransformer.dir/gemm_bias_act.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/export/users/zhaojiam/src/ByteTransformer/dpct/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object bytetransformer/src/CMakeFiles/bytetransformer.dir/gemm_bias_act.cu.o"
	cd /export/users/zhaojiam/src/ByteTransformer/dpct/bytetransformer/src && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT bytetransformer/src/CMakeFiles/bytetransformer.dir/gemm_bias_act.cu.o -MF CMakeFiles/bytetransformer.dir/gemm_bias_act.cu.o.d -x cu -c /export/users/zhaojiam/src/ByteTransformer/bytetransformer/src/gemm_bias_act.cu -o CMakeFiles/bytetransformer.dir/gemm_bias_act.cu.o

bytetransformer/src/CMakeFiles/bytetransformer.dir/gemm_bias_act.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/bytetransformer.dir/gemm_bias_act.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

bytetransformer/src/CMakeFiles/bytetransformer.dir/gemm_bias_act.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/bytetransformer.dir/gemm_bias_act.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

bytetransformer/src/CMakeFiles/bytetransformer.dir/remove_padding.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/flags.make
bytetransformer/src/CMakeFiles/bytetransformer.dir/remove_padding.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/includes_CUDA.rsp
bytetransformer/src/CMakeFiles/bytetransformer.dir/remove_padding.cu.o: /export/users/zhaojiam/src/ByteTransformer/bytetransformer/src/remove_padding.cu
bytetransformer/src/CMakeFiles/bytetransformer.dir/remove_padding.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/export/users/zhaojiam/src/ByteTransformer/dpct/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object bytetransformer/src/CMakeFiles/bytetransformer.dir/remove_padding.cu.o"
	cd /export/users/zhaojiam/src/ByteTransformer/dpct/bytetransformer/src && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT bytetransformer/src/CMakeFiles/bytetransformer.dir/remove_padding.cu.o -MF CMakeFiles/bytetransformer.dir/remove_padding.cu.o.d -x cu -c /export/users/zhaojiam/src/ByteTransformer/bytetransformer/src/remove_padding.cu -o CMakeFiles/bytetransformer.dir/remove_padding.cu.o

bytetransformer/src/CMakeFiles/bytetransformer.dir/remove_padding.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/bytetransformer.dir/remove_padding.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

bytetransformer/src/CMakeFiles/bytetransformer.dir/remove_padding.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/bytetransformer.dir/remove_padding.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

bytetransformer/src/CMakeFiles/bytetransformer.dir/layernorm.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/flags.make
bytetransformer/src/CMakeFiles/bytetransformer.dir/layernorm.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/includes_CUDA.rsp
bytetransformer/src/CMakeFiles/bytetransformer.dir/layernorm.cu.o: /export/users/zhaojiam/src/ByteTransformer/bytetransformer/src/layernorm.cu
bytetransformer/src/CMakeFiles/bytetransformer.dir/layernorm.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/export/users/zhaojiam/src/ByteTransformer/dpct/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CUDA object bytetransformer/src/CMakeFiles/bytetransformer.dir/layernorm.cu.o"
	cd /export/users/zhaojiam/src/ByteTransformer/dpct/bytetransformer/src && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT bytetransformer/src/CMakeFiles/bytetransformer.dir/layernorm.cu.o -MF CMakeFiles/bytetransformer.dir/layernorm.cu.o.d -x cu -c /export/users/zhaojiam/src/ByteTransformer/bytetransformer/src/layernorm.cu -o CMakeFiles/bytetransformer.dir/layernorm.cu.o

bytetransformer/src/CMakeFiles/bytetransformer.dir/layernorm.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/bytetransformer.dir/layernorm.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

bytetransformer/src/CMakeFiles/bytetransformer.dir/layernorm.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/bytetransformer.dir/layernorm.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_fused.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/flags.make
bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_fused.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/includes_CUDA.rsp
bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_fused.cu.o: /export/users/zhaojiam/src/ByteTransformer/bytetransformer/src/attention_fused.cu
bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_fused.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/export/users/zhaojiam/src/ByteTransformer/dpct/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CUDA object bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_fused.cu.o"
	cd /export/users/zhaojiam/src/ByteTransformer/dpct/bytetransformer/src && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_fused.cu.o -MF CMakeFiles/bytetransformer.dir/attention_fused.cu.o.d -x cu -c /export/users/zhaojiam/src/ByteTransformer/bytetransformer/src/attention_fused.cu -o CMakeFiles/bytetransformer.dir/attention_fused.cu.o

bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_fused.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/bytetransformer.dir/attention_fused.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_fused.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/bytetransformer.dir/attention_fused.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_fused_long.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/flags.make
bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_fused_long.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/includes_CUDA.rsp
bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_fused_long.cu.o: /export/users/zhaojiam/src/ByteTransformer/bytetransformer/src/attention_fused_long.cu
bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_fused_long.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/export/users/zhaojiam/src/ByteTransformer/dpct/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CUDA object bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_fused_long.cu.o"
	cd /export/users/zhaojiam/src/ByteTransformer/dpct/bytetransformer/src && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_fused_long.cu.o -MF CMakeFiles/bytetransformer.dir/attention_fused_long.cu.o.d -x cu -c /export/users/zhaojiam/src/ByteTransformer/bytetransformer/src/attention_fused_long.cu -o CMakeFiles/bytetransformer.dir/attention_fused_long.cu.o

bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_fused_long.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/bytetransformer.dir/attention_fused_long.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_fused_long.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/bytetransformer.dir/attention_fused_long.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_nofused.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/flags.make
bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_nofused.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/includes_CUDA.rsp
bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_nofused.cu.o: /export/users/zhaojiam/src/ByteTransformer/bytetransformer/src/attention_nofused.cu
bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_nofused.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/export/users/zhaojiam/src/ByteTransformer/dpct/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CUDA object bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_nofused.cu.o"
	cd /export/users/zhaojiam/src/ByteTransformer/dpct/bytetransformer/src && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_nofused.cu.o -MF CMakeFiles/bytetransformer.dir/attention_nofused.cu.o.d -x cu -c /export/users/zhaojiam/src/ByteTransformer/bytetransformer/src/attention_nofused.cu -o CMakeFiles/bytetransformer.dir/attention_nofused.cu.o

bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_nofused.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/bytetransformer.dir/attention_nofused.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_nofused.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/bytetransformer.dir/attention_nofused.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_nofused_utils.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/flags.make
bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_nofused_utils.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/includes_CUDA.rsp
bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_nofused_utils.cu.o: /export/users/zhaojiam/src/ByteTransformer/bytetransformer/src/attention_nofused_utils.cu
bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_nofused_utils.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/export/users/zhaojiam/src/ByteTransformer/dpct/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CUDA object bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_nofused_utils.cu.o"
	cd /export/users/zhaojiam/src/ByteTransformer/dpct/bytetransformer/src && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_nofused_utils.cu.o -MF CMakeFiles/bytetransformer.dir/attention_nofused_utils.cu.o.d -x cu -c /export/users/zhaojiam/src/ByteTransformer/bytetransformer/src/attention_nofused_utils.cu -o CMakeFiles/bytetransformer.dir/attention_nofused_utils.cu.o

bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_nofused_utils.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/bytetransformer.dir/attention_nofused_utils.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_nofused_utils.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/bytetransformer.dir/attention_nofused_utils.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

bytetransformer/src/CMakeFiles/bytetransformer.dir/bert_transformer.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/flags.make
bytetransformer/src/CMakeFiles/bytetransformer.dir/bert_transformer.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/includes_CUDA.rsp
bytetransformer/src/CMakeFiles/bytetransformer.dir/bert_transformer.cu.o: /export/users/zhaojiam/src/ByteTransformer/bytetransformer/src/bert_transformer.cu
bytetransformer/src/CMakeFiles/bytetransformer.dir/bert_transformer.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/export/users/zhaojiam/src/ByteTransformer/dpct/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CUDA object bytetransformer/src/CMakeFiles/bytetransformer.dir/bert_transformer.cu.o"
	cd /export/users/zhaojiam/src/ByteTransformer/dpct/bytetransformer/src && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT bytetransformer/src/CMakeFiles/bytetransformer.dir/bert_transformer.cu.o -MF CMakeFiles/bytetransformer.dir/bert_transformer.cu.o.d -x cu -c /export/users/zhaojiam/src/ByteTransformer/bytetransformer/src/bert_transformer.cu -o CMakeFiles/bytetransformer.dir/bert_transformer.cu.o

bytetransformer/src/CMakeFiles/bytetransformer.dir/bert_transformer.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/bytetransformer.dir/bert_transformer.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

bytetransformer/src/CMakeFiles/bytetransformer.dir/bert_transformer.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/bytetransformer.dir/bert_transformer.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

bytetransformer/src/CMakeFiles/bytetransformer.dir/cutlass_attention.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/flags.make
bytetransformer/src/CMakeFiles/bytetransformer.dir/cutlass_attention.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/includes_CUDA.rsp
bytetransformer/src/CMakeFiles/bytetransformer.dir/cutlass_attention.cu.o: /export/users/zhaojiam/src/ByteTransformer/bytetransformer/src/cutlass_attention.cu
bytetransformer/src/CMakeFiles/bytetransformer.dir/cutlass_attention.cu.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/export/users/zhaojiam/src/ByteTransformer/dpct/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CUDA object bytetransformer/src/CMakeFiles/bytetransformer.dir/cutlass_attention.cu.o"
	cd /export/users/zhaojiam/src/ByteTransformer/dpct/bytetransformer/src && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) --use_fast_math -MD -MT bytetransformer/src/CMakeFiles/bytetransformer.dir/cutlass_attention.cu.o -MF CMakeFiles/bytetransformer.dir/cutlass_attention.cu.o.d -x cu -c /export/users/zhaojiam/src/ByteTransformer/bytetransformer/src/cutlass_attention.cu -o CMakeFiles/bytetransformer.dir/cutlass_attention.cu.o

bytetransformer/src/CMakeFiles/bytetransformer.dir/cutlass_attention.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/bytetransformer.dir/cutlass_attention.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

bytetransformer/src/CMakeFiles/bytetransformer.dir/cutlass_attention.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/bytetransformer.dir/cutlass_attention.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target bytetransformer
bytetransformer_OBJECTS = \
"CMakeFiles/bytetransformer.dir/gemm.cu.o" \
"CMakeFiles/bytetransformer.dir/gemm_bias_act.cu.o" \
"CMakeFiles/bytetransformer.dir/remove_padding.cu.o" \
"CMakeFiles/bytetransformer.dir/layernorm.cu.o" \
"CMakeFiles/bytetransformer.dir/attention_fused.cu.o" \
"CMakeFiles/bytetransformer.dir/attention_fused_long.cu.o" \
"CMakeFiles/bytetransformer.dir/attention_nofused.cu.o" \
"CMakeFiles/bytetransformer.dir/attention_nofused_utils.cu.o" \
"CMakeFiles/bytetransformer.dir/bert_transformer.cu.o" \
"CMakeFiles/bytetransformer.dir/cutlass_attention.cu.o"

# External object files for target bytetransformer
bytetransformer_EXTERNAL_OBJECTS =

bytetransformer/src/CMakeFiles/bytetransformer.dir/cmake_device_link.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/gemm.cu.o
bytetransformer/src/CMakeFiles/bytetransformer.dir/cmake_device_link.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/gemm_bias_act.cu.o
bytetransformer/src/CMakeFiles/bytetransformer.dir/cmake_device_link.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/remove_padding.cu.o
bytetransformer/src/CMakeFiles/bytetransformer.dir/cmake_device_link.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/layernorm.cu.o
bytetransformer/src/CMakeFiles/bytetransformer.dir/cmake_device_link.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_fused.cu.o
bytetransformer/src/CMakeFiles/bytetransformer.dir/cmake_device_link.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_fused_long.cu.o
bytetransformer/src/CMakeFiles/bytetransformer.dir/cmake_device_link.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_nofused.cu.o
bytetransformer/src/CMakeFiles/bytetransformer.dir/cmake_device_link.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_nofused_utils.cu.o
bytetransformer/src/CMakeFiles/bytetransformer.dir/cmake_device_link.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/bert_transformer.cu.o
bytetransformer/src/CMakeFiles/bytetransformer.dir/cmake_device_link.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/cutlass_attention.cu.o
bytetransformer/src/CMakeFiles/bytetransformer.dir/cmake_device_link.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/build.make
bytetransformer/src/CMakeFiles/bytetransformer.dir/cmake_device_link.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/deviceLinkLibs.rsp
bytetransformer/src/CMakeFiles/bytetransformer.dir/cmake_device_link.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/deviceObjects1.rsp
bytetransformer/src/CMakeFiles/bytetransformer.dir/cmake_device_link.o: bytetransformer/src/CMakeFiles/bytetransformer.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/export/users/zhaojiam/src/ByteTransformer/dpct/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Linking CUDA device code CMakeFiles/bytetransformer.dir/cmake_device_link.o"
	cd /export/users/zhaojiam/src/ByteTransformer/dpct/bytetransformer/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bytetransformer.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
bytetransformer/src/CMakeFiles/bytetransformer.dir/build: bytetransformer/src/CMakeFiles/bytetransformer.dir/cmake_device_link.o
.PHONY : bytetransformer/src/CMakeFiles/bytetransformer.dir/build

# Object files for target bytetransformer
bytetransformer_OBJECTS = \
"CMakeFiles/bytetransformer.dir/gemm.cu.o" \
"CMakeFiles/bytetransformer.dir/gemm_bias_act.cu.o" \
"CMakeFiles/bytetransformer.dir/remove_padding.cu.o" \
"CMakeFiles/bytetransformer.dir/layernorm.cu.o" \
"CMakeFiles/bytetransformer.dir/attention_fused.cu.o" \
"CMakeFiles/bytetransformer.dir/attention_fused_long.cu.o" \
"CMakeFiles/bytetransformer.dir/attention_nofused.cu.o" \
"CMakeFiles/bytetransformer.dir/attention_nofused_utils.cu.o" \
"CMakeFiles/bytetransformer.dir/bert_transformer.cu.o" \
"CMakeFiles/bytetransformer.dir/cutlass_attention.cu.o"

# External object files for target bytetransformer
bytetransformer_EXTERNAL_OBJECTS =

lib/libbytetransformer.so: bytetransformer/src/CMakeFiles/bytetransformer.dir/gemm.cu.o
lib/libbytetransformer.so: bytetransformer/src/CMakeFiles/bytetransformer.dir/gemm_bias_act.cu.o
lib/libbytetransformer.so: bytetransformer/src/CMakeFiles/bytetransformer.dir/remove_padding.cu.o
lib/libbytetransformer.so: bytetransformer/src/CMakeFiles/bytetransformer.dir/layernorm.cu.o
lib/libbytetransformer.so: bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_fused.cu.o
lib/libbytetransformer.so: bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_fused_long.cu.o
lib/libbytetransformer.so: bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_nofused.cu.o
lib/libbytetransformer.so: bytetransformer/src/CMakeFiles/bytetransformer.dir/attention_nofused_utils.cu.o
lib/libbytetransformer.so: bytetransformer/src/CMakeFiles/bytetransformer.dir/bert_transformer.cu.o
lib/libbytetransformer.so: bytetransformer/src/CMakeFiles/bytetransformer.dir/cutlass_attention.cu.o
lib/libbytetransformer.so: bytetransformer/src/CMakeFiles/bytetransformer.dir/build.make
lib/libbytetransformer.so: bytetransformer/src/CMakeFiles/bytetransformer.dir/cmake_device_link.o
lib/libbytetransformer.so: bytetransformer/src/CMakeFiles/bytetransformer.dir/linkLibs.rsp
lib/libbytetransformer.so: bytetransformer/src/CMakeFiles/bytetransformer.dir/objects1.rsp
lib/libbytetransformer.so: bytetransformer/src/CMakeFiles/bytetransformer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/export/users/zhaojiam/src/ByteTransformer/dpct/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Linking CUDA shared library ../../lib/libbytetransformer.so"
	cd /export/users/zhaojiam/src/ByteTransformer/dpct/bytetransformer/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bytetransformer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
bytetransformer/src/CMakeFiles/bytetransformer.dir/build: lib/libbytetransformer.so
.PHONY : bytetransformer/src/CMakeFiles/bytetransformer.dir/build

bytetransformer/src/CMakeFiles/bytetransformer.dir/clean:
	cd /export/users/zhaojiam/src/ByteTransformer/dpct/bytetransformer/src && $(CMAKE_COMMAND) -P CMakeFiles/bytetransformer.dir/cmake_clean.cmake
.PHONY : bytetransformer/src/CMakeFiles/bytetransformer.dir/clean

bytetransformer/src/CMakeFiles/bytetransformer.dir/depend:
	cd /export/users/zhaojiam/src/ByteTransformer/dpct && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /export/users/zhaojiam/src/ByteTransformer /export/users/zhaojiam/src/ByteTransformer/bytetransformer/src /export/users/zhaojiam/src/ByteTransformer/dpct /export/users/zhaojiam/src/ByteTransformer/dpct/bytetransformer/src /export/users/zhaojiam/src/ByteTransformer/dpct/bytetransformer/src/CMakeFiles/bytetransformer.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : bytetransformer/src/CMakeFiles/bytetransformer.dir/depend

