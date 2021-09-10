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
CMAKE_COMMAND = /home/smooth_operator/programs/CLion/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/smooth_operator/programs/CLion/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/smooth_operator/fun/euler/pr30_digit_fifth_powers

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/smooth_operator/fun/euler/pr30_digit_fifth_powers/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/pr30_digit_fifth_powers.dir/depend.make
# Include the progress variables for this target.
include CMakeFiles/pr30_digit_fifth_powers.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pr30_digit_fifth_powers.dir/flags.make

CMakeFiles/pr30_digit_fifth_powers.dir/main.cu.o: CMakeFiles/pr30_digit_fifth_powers.dir/flags.make
CMakeFiles/pr30_digit_fifth_powers.dir/main.cu.o: ../main.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/smooth_operator/fun/euler/pr30_digit_fifth_powers/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/pr30_digit_fifth_powers.dir/main.cu.o"
	/opt/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/smooth_operator/fun/euler/pr30_digit_fifth_powers/main.cu -o CMakeFiles/pr30_digit_fifth_powers.dir/main.cu.o

CMakeFiles/pr30_digit_fifth_powers.dir/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/pr30_digit_fifth_powers.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/pr30_digit_fifth_powers.dir/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/pr30_digit_fifth_powers.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target pr30_digit_fifth_powers
pr30_digit_fifth_powers_OBJECTS = \
"CMakeFiles/pr30_digit_fifth_powers.dir/main.cu.o"

# External object files for target pr30_digit_fifth_powers
pr30_digit_fifth_powers_EXTERNAL_OBJECTS =

CMakeFiles/pr30_digit_fifth_powers.dir/cmake_device_link.o: CMakeFiles/pr30_digit_fifth_powers.dir/main.cu.o
CMakeFiles/pr30_digit_fifth_powers.dir/cmake_device_link.o: CMakeFiles/pr30_digit_fifth_powers.dir/build.make
CMakeFiles/pr30_digit_fifth_powers.dir/cmake_device_link.o: CMakeFiles/pr30_digit_fifth_powers.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/smooth_operator/fun/euler/pr30_digit_fifth_powers/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/pr30_digit_fifth_powers.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pr30_digit_fifth_powers.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pr30_digit_fifth_powers.dir/build: CMakeFiles/pr30_digit_fifth_powers.dir/cmake_device_link.o
.PHONY : CMakeFiles/pr30_digit_fifth_powers.dir/build

# Object files for target pr30_digit_fifth_powers
pr30_digit_fifth_powers_OBJECTS = \
"CMakeFiles/pr30_digit_fifth_powers.dir/main.cu.o"

# External object files for target pr30_digit_fifth_powers
pr30_digit_fifth_powers_EXTERNAL_OBJECTS =

pr30_digit_fifth_powers: CMakeFiles/pr30_digit_fifth_powers.dir/main.cu.o
pr30_digit_fifth_powers: CMakeFiles/pr30_digit_fifth_powers.dir/build.make
pr30_digit_fifth_powers: CMakeFiles/pr30_digit_fifth_powers.dir/cmake_device_link.o
pr30_digit_fifth_powers: CMakeFiles/pr30_digit_fifth_powers.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/smooth_operator/fun/euler/pr30_digit_fifth_powers/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable pr30_digit_fifth_powers"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pr30_digit_fifth_powers.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pr30_digit_fifth_powers.dir/build: pr30_digit_fifth_powers
.PHONY : CMakeFiles/pr30_digit_fifth_powers.dir/build

CMakeFiles/pr30_digit_fifth_powers.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pr30_digit_fifth_powers.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pr30_digit_fifth_powers.dir/clean

CMakeFiles/pr30_digit_fifth_powers.dir/depend:
	cd /home/smooth_operator/fun/euler/pr30_digit_fifth_powers/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/smooth_operator/fun/euler/pr30_digit_fifth_powers /home/smooth_operator/fun/euler/pr30_digit_fifth_powers /home/smooth_operator/fun/euler/pr30_digit_fifth_powers/cmake-build-debug /home/smooth_operator/fun/euler/pr30_digit_fifth_powers/cmake-build-debug /home/smooth_operator/fun/euler/pr30_digit_fifth_powers/cmake-build-debug/CMakeFiles/pr30_digit_fifth_powers.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pr30_digit_fifth_powers.dir/depend

