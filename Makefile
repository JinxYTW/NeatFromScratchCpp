# The Makefile must be in the root directory of the project where the source files are located.
#
# Usage (in a terminal in the root directory of the project):
# make            # Compile and link all .cpp files
# make clean      # Clean up the build directory

# Compiler and linker - Use g++ on Linux, Windows and clang++ on Mac OS X
CXX        = g++
# Compiler options - Wall for all warnings, std=c++17 for C++17
CXXFLAGS   = -Wall -std=c++17
# Dependency flags - Include .d files generated by the compiler
DEPFLAGS   = -MMD
# Linker flags - No flags
LDFLAGS    = 
# Build directory
BUILDIR    = build
# Source files - All .cpp files required to build the executable
SRC_FILES  = mainrpcshow.cpp ComputeFitness.cpp Genome.cpp population.cpp GenomeIndexer.cpp neat.cpp NeuralNetwork.cpp Utils.cpp LayerManager.cpp Mutator.cpp 
# Object files - All .o files generated from the source files
OBJ_FILES  = $(patsubst %.cpp, $(BUILDIR)/%.o, $(SRC_FILES))
# Executable - The name of the executable into the bin directory
BINDIR     = bin
# Target - The path to the executable
TARGET     = $(BINDIR)/app
# Dependencies - All .d files generated by the compiler
DEPS       = $(OBJ_FILES:.o=.d)

all: $(TARGET)

# Link object files to executable into the bin directory
$(TARGET): $(OBJ_FILES)
	@if not exist $(BINDIR) mkdir $(BINDIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) $(OBJ_FILES)

# Compile source files to object files into the build directory
$(BUILDIR)/%.o: %.cpp
	@if not exist $(BUILDIR) mkdir $(BUILDIR)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDE) -c $< -o $@

# Clean up the build and bin directories
clean:
	-del /Q /S $(BUILDIR) $(BINDIR) 2>nul

# Include the dependencies generated by the compiler
-include $(DEPS)

# Phony targets
.PHONY: all clean
