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
CMAKE_SOURCE_DIR = /home/smooth_operator/fun/euler/pr20_factorial_sum

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/smooth_operator/fun/euler/pr20_factorial_sum/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/pr20_factorial_sum.dir/depend.make
# Include the progress variables for this target.
include CMakeFiles/pr20_factorial_sum.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pr20_factorial_sum.dir/flags.make

CMakeFiles/pr20_factorial_sum.dir/main.cpp.o: CMakeFiles/pr20_factorial_sum.dir/flags.make
CMakeFiles/pr20_factorial_sum.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/smooth_operator/fun/euler/pr20_factorial_sum/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pr20_factorial_sum.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pr20_factorial_sum.dir/main.cpp.o -c /home/smooth_operator/fun/euler/pr20_factorial_sum/main.cpp

CMakeFiles/pr20_factorial_sum.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pr20_factorial_sum.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/smooth_operator/fun/euler/pr20_factorial_sum/main.cpp > CMakeFiles/pr20_factorial_sum.dir/main.cpp.i

CMakeFiles/pr20_factorial_sum.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pr20_factorial_sum.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/smooth_operator/fun/euler/pr20_factorial_sum/main.cpp -o CMakeFiles/pr20_factorial_sum.dir/main.cpp.s

# Object files for target pr20_factorial_sum
pr20_factorial_sum_OBJECTS = \
"CMakeFiles/pr20_factorial_sum.dir/main.cpp.o"

# External object files for target pr20_factorial_sum
pr20_factorial_sum_EXTERNAL_OBJECTS =

pr20_factorial_sum: CMakeFiles/pr20_factorial_sum.dir/main.cpp.o
pr20_factorial_sum: CMakeFiles/pr20_factorial_sum.dir/build.make
pr20_factorial_sum: CMakeFiles/pr20_factorial_sum.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/smooth_operator/fun/euler/pr20_factorial_sum/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pr20_factorial_sum"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pr20_factorial_sum.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pr20_factorial_sum.dir/build: pr20_factorial_sum
.PHONY : CMakeFiles/pr20_factorial_sum.dir/build

CMakeFiles/pr20_factorial_sum.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pr20_factorial_sum.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pr20_factorial_sum.dir/clean

CMakeFiles/pr20_factorial_sum.dir/depend:
	cd /home/smooth_operator/fun/euler/pr20_factorial_sum/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/smooth_operator/fun/euler/pr20_factorial_sum /home/smooth_operator/fun/euler/pr20_factorial_sum /home/smooth_operator/fun/euler/pr20_factorial_sum/cmake-build-debug /home/smooth_operator/fun/euler/pr20_factorial_sum/cmake-build-debug /home/smooth_operator/fun/euler/pr20_factorial_sum/cmake-build-debug/CMakeFiles/pr20_factorial_sum.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pr20_factorial_sum.dir/depend

