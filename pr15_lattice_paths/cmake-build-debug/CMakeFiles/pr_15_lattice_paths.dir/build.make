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
CMAKE_SOURCE_DIR = /home/smooth_operator/fun/euler/pr15_lattice_paths

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/smooth_operator/fun/euler/pr15_lattice_paths/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/pr_15_lattice_paths.dir/depend.make
# Include the progress variables for this target.
include CMakeFiles/pr_15_lattice_paths.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pr_15_lattice_paths.dir/flags.make

CMakeFiles/pr_15_lattice_paths.dir/main.cpp.o: CMakeFiles/pr_15_lattice_paths.dir/flags.make
CMakeFiles/pr_15_lattice_paths.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/smooth_operator/fun/euler/pr15_lattice_paths/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pr_15_lattice_paths.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pr_15_lattice_paths.dir/main.cpp.o -c /home/smooth_operator/fun/euler/pr15_lattice_paths/main.cpp

CMakeFiles/pr_15_lattice_paths.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pr_15_lattice_paths.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/smooth_operator/fun/euler/pr15_lattice_paths/main.cpp > CMakeFiles/pr_15_lattice_paths.dir/main.cpp.i

CMakeFiles/pr_15_lattice_paths.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pr_15_lattice_paths.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/smooth_operator/fun/euler/pr15_lattice_paths/main.cpp -o CMakeFiles/pr_15_lattice_paths.dir/main.cpp.s

# Object files for target pr_15_lattice_paths
pr_15_lattice_paths_OBJECTS = \
"CMakeFiles/pr_15_lattice_paths.dir/main.cpp.o"

# External object files for target pr_15_lattice_paths
pr_15_lattice_paths_EXTERNAL_OBJECTS =

pr_15_lattice_paths: CMakeFiles/pr_15_lattice_paths.dir/main.cpp.o
pr_15_lattice_paths: CMakeFiles/pr_15_lattice_paths.dir/build.make
pr_15_lattice_paths: CMakeFiles/pr_15_lattice_paths.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/smooth_operator/fun/euler/pr15_lattice_paths/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pr_15_lattice_paths"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pr_15_lattice_paths.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pr_15_lattice_paths.dir/build: pr_15_lattice_paths
.PHONY : CMakeFiles/pr_15_lattice_paths.dir/build

CMakeFiles/pr_15_lattice_paths.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pr_15_lattice_paths.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pr_15_lattice_paths.dir/clean

CMakeFiles/pr_15_lattice_paths.dir/depend:
	cd /home/smooth_operator/fun/euler/pr15_lattice_paths/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/smooth_operator/fun/euler/pr15_lattice_paths /home/smooth_operator/fun/euler/pr15_lattice_paths /home/smooth_operator/fun/euler/pr15_lattice_paths/cmake-build-debug /home/smooth_operator/fun/euler/pr15_lattice_paths/cmake-build-debug /home/smooth_operator/fun/euler/pr15_lattice_paths/cmake-build-debug/CMakeFiles/pr_15_lattice_paths.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pr_15_lattice_paths.dir/depend
