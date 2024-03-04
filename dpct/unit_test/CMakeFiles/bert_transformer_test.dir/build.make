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
include unit_test/CMakeFiles/bert_transformer_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include unit_test/CMakeFiles/bert_transformer_test.dir/compiler_depend.make

# Include the progress variables for this target.
include unit_test/CMakeFiles/bert_transformer_test.dir/progress.make

# Include the compile flags for this target's objects.
include unit_test/CMakeFiles/bert_transformer_test.dir/flags.make

unit_test/CMakeFiles/bert_transformer_test.dir/bert_transformer_test.cc.o: unit_test/CMakeFiles/bert_transformer_test.dir/flags.make
unit_test/CMakeFiles/bert_transformer_test.dir/bert_transformer_test.cc.o: /export/users/zhaojiam/src/ByteTransformer/unit_test/bert_transformer_test.cc
unit_test/CMakeFiles/bert_transformer_test.dir/bert_transformer_test.cc.o: unit_test/CMakeFiles/bert_transformer_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/export/users/zhaojiam/src/ByteTransformer/dpct/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object unit_test/CMakeFiles/bert_transformer_test.dir/bert_transformer_test.cc.o"
	cd /export/users/zhaojiam/src/ByteTransformer/dpct/unit_test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT unit_test/CMakeFiles/bert_transformer_test.dir/bert_transformer_test.cc.o -MF CMakeFiles/bert_transformer_test.dir/bert_transformer_test.cc.o.d -o CMakeFiles/bert_transformer_test.dir/bert_transformer_test.cc.o -c /export/users/zhaojiam/src/ByteTransformer/unit_test/bert_transformer_test.cc

unit_test/CMakeFiles/bert_transformer_test.dir/bert_transformer_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/bert_transformer_test.dir/bert_transformer_test.cc.i"
	cd /export/users/zhaojiam/src/ByteTransformer/dpct/unit_test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /export/users/zhaojiam/src/ByteTransformer/unit_test/bert_transformer_test.cc > CMakeFiles/bert_transformer_test.dir/bert_transformer_test.cc.i

unit_test/CMakeFiles/bert_transformer_test.dir/bert_transformer_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/bert_transformer_test.dir/bert_transformer_test.cc.s"
	cd /export/users/zhaojiam/src/ByteTransformer/dpct/unit_test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /export/users/zhaojiam/src/ByteTransformer/unit_test/bert_transformer_test.cc -o CMakeFiles/bert_transformer_test.dir/bert_transformer_test.cc.s

# Object files for target bert_transformer_test
bert_transformer_test_OBJECTS = \
"CMakeFiles/bert_transformer_test.dir/bert_transformer_test.cc.o"

# External object files for target bert_transformer_test
bert_transformer_test_EXTERNAL_OBJECTS =

bin/bert_transformer_test: unit_test/CMakeFiles/bert_transformer_test.dir/bert_transformer_test.cc.o
bin/bert_transformer_test: unit_test/CMakeFiles/bert_transformer_test.dir/build.make
bin/bert_transformer_test: lib/libbytetransformer.so
bin/bert_transformer_test: unit_test/CMakeFiles/bert_transformer_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/export/users/zhaojiam/src/ByteTransformer/dpct/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/bert_transformer_test"
	cd /export/users/zhaojiam/src/ByteTransformer/dpct/unit_test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bert_transformer_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
unit_test/CMakeFiles/bert_transformer_test.dir/build: bin/bert_transformer_test
.PHONY : unit_test/CMakeFiles/bert_transformer_test.dir/build

unit_test/CMakeFiles/bert_transformer_test.dir/clean:
	cd /export/users/zhaojiam/src/ByteTransformer/dpct/unit_test && $(CMAKE_COMMAND) -P CMakeFiles/bert_transformer_test.dir/cmake_clean.cmake
.PHONY : unit_test/CMakeFiles/bert_transformer_test.dir/clean

unit_test/CMakeFiles/bert_transformer_test.dir/depend:
	cd /export/users/zhaojiam/src/ByteTransformer/dpct && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /export/users/zhaojiam/src/ByteTransformer /export/users/zhaojiam/src/ByteTransformer/unit_test /export/users/zhaojiam/src/ByteTransformer/dpct /export/users/zhaojiam/src/ByteTransformer/dpct/unit_test /export/users/zhaojiam/src/ByteTransformer/dpct/unit_test/CMakeFiles/bert_transformer_test.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : unit_test/CMakeFiles/bert_transformer_test.dir/depend

