# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

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
CMAKE_COMMAND = /home2/chufansh/.local/cmake-3.20.3-linux-x86_64/bin/cmake

# The command to remove a file.
RM = /home2/chufansh/.local/cmake-3.20.3-linux-x86_64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home2/chufansh/dpfile/SparseExt

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home2/chufansh/dpfile/SparseExt/build

# Include any dependencies generated for this target.
include src/CMakeFiles/_C.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/_C.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/_C.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/_C.dir/flags.make

src/CMakeFiles/_C.dir/init_binding.cpp.o: src/CMakeFiles/_C.dir/flags.make
src/CMakeFiles/_C.dir/init_binding.cpp.o: ../src/init_binding.cpp
src/CMakeFiles/_C.dir/init_binding.cpp.o: src/CMakeFiles/_C.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home2/chufansh/dpfile/SparseExt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/_C.dir/init_binding.cpp.o"
	cd /home2/chufansh/dpfile/SparseExt/build/src && /opt/rh/devtoolset-9/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/_C.dir/init_binding.cpp.o -MF CMakeFiles/_C.dir/init_binding.cpp.o.d -o CMakeFiles/_C.dir/init_binding.cpp.o -c /home2/chufansh/dpfile/SparseExt/src/init_binding.cpp

src/CMakeFiles/_C.dir/init_binding.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/_C.dir/init_binding.cpp.i"
	cd /home2/chufansh/dpfile/SparseExt/build/src && /opt/rh/devtoolset-9/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home2/chufansh/dpfile/SparseExt/src/init_binding.cpp > CMakeFiles/_C.dir/init_binding.cpp.i

src/CMakeFiles/_C.dir/init_binding.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/_C.dir/init_binding.cpp.s"
	cd /home2/chufansh/dpfile/SparseExt/build/src && /opt/rh/devtoolset-9/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home2/chufansh/dpfile/SparseExt/src/init_binding.cpp -o CMakeFiles/_C.dir/init_binding.cpp.s

# Object files for target _C
_C_OBJECTS = \
"CMakeFiles/_C.dir/init_binding.cpp.o"

# External object files for target _C
_C_EXTERNAL_OBJECTS =

lib.linux-x86_64-3.8/_C.cpython-38-x86_64-linux-gnu.so: src/CMakeFiles/_C.dir/init_binding.cpp.o
lib.linux-x86_64-3.8/_C.cpython-38-x86_64-linux-gnu.so: src/CMakeFiles/_C.dir/build.make
lib.linux-x86_64-3.8/_C.cpython-38-x86_64-linux-gnu.so: /home2/chufansh/anaconda3/envs/pytorch/lib/python3.8/site-packages/torch/lib/libtorch.so
lib.linux-x86_64-3.8/_C.cpython-38-x86_64-linux-gnu.so: /home2/chufansh/anaconda3/envs/pytorch/lib/python3.8/site-packages/torch/lib/libc10.so
lib.linux-x86_64-3.8/_C.cpython-38-x86_64-linux-gnu.so: /home2/chufansh/anaconda3/envs/pytorch/lib/python3.8/site-packages/torch/lib/libtorch_python.so
lib.linux-x86_64-3.8/_C.cpython-38-x86_64-linux-gnu.so: /home2/chufansh/anaconda3/envs/pytorch/lib/python3.8/site-packages/torch/lib/libc10.so
lib.linux-x86_64-3.8/_C.cpython-38-x86_64-linux-gnu.so: src/CMakeFiles/_C.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home2/chufansh/dpfile/SparseExt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module ../lib.linux-x86_64-3.8/_C.cpython-38-x86_64-linux-gnu.so"
	cd /home2/chufansh/dpfile/SparseExt/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/_C.dir/link.txt --verbose=$(VERBOSE)
	cd /home2/chufansh/dpfile/SparseExt/build/src && /opt/rh/devtoolset-9/root/usr/bin/strip /home2/chufansh/dpfile/SparseExt/build/lib.linux-x86_64-3.8/_C.cpython-38-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
src/CMakeFiles/_C.dir/build: lib.linux-x86_64-3.8/_C.cpython-38-x86_64-linux-gnu.so
.PHONY : src/CMakeFiles/_C.dir/build

src/CMakeFiles/_C.dir/clean:
	cd /home2/chufansh/dpfile/SparseExt/build/src && $(CMAKE_COMMAND) -P CMakeFiles/_C.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/_C.dir/clean

src/CMakeFiles/_C.dir/depend:
	cd /home2/chufansh/dpfile/SparseExt/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home2/chufansh/dpfile/SparseExt /home2/chufansh/dpfile/SparseExt/src /home2/chufansh/dpfile/SparseExt/build /home2/chufansh/dpfile/SparseExt/build/src /home2/chufansh/dpfile/SparseExt/build/src/CMakeFiles/_C.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/_C.dir/depend

