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

# Utility rule file for copy_ut_scripts.

# Include any custom commands dependencies for this target.
include unit_test/CMakeFiles/copy_ut_scripts.dir/compiler_depend.make

# Include the progress variables for this target.
include unit_test/CMakeFiles/copy_ut_scripts.dir/progress.make

copy_ut_scripts: unit_test/CMakeFiles/copy_ut_scripts.dir/build.make
	cd /export/users/zhaojiam/src/ByteTransformer/dpct/unit_test && cp /export/users/zhaojiam/src/ByteTransformer/unit_test/python_scripts/*.py /export/users/zhaojiam/src/ByteTransformer/dpct
.PHONY : copy_ut_scripts

# Rule to build all files generated by this target.
unit_test/CMakeFiles/copy_ut_scripts.dir/build: copy_ut_scripts
.PHONY : unit_test/CMakeFiles/copy_ut_scripts.dir/build

unit_test/CMakeFiles/copy_ut_scripts.dir/clean:
	cd /export/users/zhaojiam/src/ByteTransformer/dpct/unit_test && $(CMAKE_COMMAND) -P CMakeFiles/copy_ut_scripts.dir/cmake_clean.cmake
.PHONY : unit_test/CMakeFiles/copy_ut_scripts.dir/clean

unit_test/CMakeFiles/copy_ut_scripts.dir/depend:
	cd /export/users/zhaojiam/src/ByteTransformer/dpct && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /export/users/zhaojiam/src/ByteTransformer /export/users/zhaojiam/src/ByteTransformer/unit_test /export/users/zhaojiam/src/ByteTransformer/dpct /export/users/zhaojiam/src/ByteTransformer/dpct/unit_test /export/users/zhaojiam/src/ByteTransformer/dpct/unit_test/CMakeFiles/copy_ut_scripts.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : unit_test/CMakeFiles/copy_ut_scripts.dir/depend
