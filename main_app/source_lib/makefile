##############################################################################
# source_lib Makefile
# This file builds the dependency static libraries (BayesClassifier and
# BayesPipeline) via CMake, then compiles source_lib sources and links
# everything together.
##############################################################################

# C++ compiler executable used for compile and link steps.
CXX      := g++
# Global compile flags:
# -std=c++17 : language level
# -Wall/-Wextra : extra diagnostics
# -fPIC : position-independent code for shared library linking
CXXFLAGS := -std=c++17 -Wall -Wextra -fPIC

# Root source directory for this project.
SRCDIR   := cppsource
# Output directories for generated artifacts.
BUILDDIR := build
OBJDIR   := $(BUILDDIR)/obj

# BayesClassifier paths.
BAYES_SRC_DIR     := $(CURDIR)/deps/BayesClassifier
BAYES_LIB_DIR     := $(CURDIR)/build/bayes
BAYES_INCLUDE_DIR := $(BAYES_SRC_DIR)/include

# BayesPipeline paths.
BAYES_PIPELINE_DIR := $(CURDIR)/deps/BayesPipeline
BAYES_PIPELINE_BUILD_DIR := $(BAYES_PIPELINE_DIR)/build

# Static library artifacts produced by CMake.
BAYES_CORE_LIB    := $(BAYES_LIB_DIR)/libnaive_bayes_core.a
BAYES_PIPELINE_LIB := $(BAYES_PIPELINE_BUILD_DIR)/libbayes_pipeline.a

# Header include flags shared by all compilation units.
# Include app source headers, BayesClassifier headers, and BayesPipeline headers.
INCLUDES := -I$(SRCDIR) \
            -I$(BAYES_INCLUDE_DIR) \
            -I$(BAYES_PIPELINE_DIR)/include

# Convenience variable for static libs + system link libs.
# Used by target-specific link commands as needed.
LIBS := $(BAYES_PIPELINE_DIR)/build/libbayes_pipeline.a \
        $(BAYES_LIB_DIR)/libnaive_bayes_core.a \
        -ldl -lpthread

# Object files that comprise the shared library and most of the executable.
# These come from source_lib code; BayesPipeline code is linked from static lib.
LIB_OBJS += $(OBJDIR)/main_app.o
LIB_OBJS += $(OBJDIR)/MessageSimulator.o
LIB_OBJS += $(OBJDIR)/TestMessageProcessor1.o
LIB_OBJS += $(OBJDIR)/TestMessageProcessor2.o

# Output shared library name.
TARGETLIB := $(BUILDDIR)/libsource_lib.so
# Output executable name.
TARGETAPP := $(BUILDDIR)/main_app
# Output deterministic scenario test executable.
TARGETSCENARIO := $(BUILDDIR)/scenario_tests

# Declare pseudo-targets that do not map to real files.
.PHONY: all clean DIRS main_app scenario_tests libsource_lib deps

# Default build target: build deps, then produce shared library and executable.
all: $(TARGETLIB) $(TARGETAPP)

##############################################################################
# Dependency builds via CMake
##############################################################################

# Build BayesClassifier static library via CMake.
$(BAYES_CORE_LIB):
	@mkdir -p $(BAYES_LIB_DIR)
	cmake -S $(BAYES_SRC_DIR) -B $(BAYES_LIB_DIR) \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
	cmake --build $(BAYES_LIB_DIR) --target naive_bayes_core --config Release

# Build BayesPipeline static library via CMake (depends on BayesClassifier).
$(BAYES_PIPELINE_LIB): $(BAYES_CORE_LIB)
	@mkdir -p $(BAYES_PIPELINE_BUILD_DIR)
	cmake -S $(BAYES_PIPELINE_DIR) -B $(BAYES_PIPELINE_BUILD_DIR) \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
	    -DBAYES_INCLUDE_DIR=$(BAYES_INCLUDE_DIR)
	cmake --build $(BAYES_PIPELINE_BUILD_DIR) --config Release

# Convenience target to build just the dependencies.
deps: $(BAYES_CORE_LIB) $(BAYES_PIPELINE_LIB)

# Compatibility aliases to keep old target names working.
main_app: $(TARGETAPP)

scenario_tests: $(TARGETSCENARIO)

libsource_lib: $(TARGETLIB)

# Utility target: ensure object directory exists before compiling.
DIRS:
	@mkdir -p $(BUILDDIR) $(OBJDIR)

# Link shared library from local objects plus both static dependency libs.
# whole-archive is used so required objects are not discarded by the linker.
$(TARGETLIB): DIRS $(LIB_OBJS) $(BAYES_PIPELINE_LIB) $(BAYES_CORE_LIB)
	$(CXX) $(CXXFLAGS) -shared -o $@ $(LIB_OBJS) \
	    -Wl,--whole-archive $(BAYES_PIPELINE_LIB) $(BAYES_CORE_LIB) \
	    -Wl,--no-whole-archive \
	    -ldl -lpthread
	@echo "Built $(TARGETLIB)"

# Link executable from main object + shared local objects + static dependency libs.
# whole-archive is also used here to retain required static-library object files.
$(TARGETAPP): DIRS $(OBJDIR)/main.o $(LIB_OBJS) $(BAYES_PIPELINE_LIB) $(BAYES_CORE_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJDIR)/main.o $(LIB_OBJS) \
	    -Wl,--whole-archive $(BAYES_PIPELINE_LIB) $(BAYES_CORE_LIB) \
	    -Wl,--no-whole-archive \
	    -ldl -lpthread
	@echo "Built $(TARGETAPP)  →  run with:  ./$(TARGETAPP)"

# Link deterministic scenario test executable.
$(TARGETSCENARIO): DIRS $(OBJDIR)/scenario_tests.o $(OBJDIR)/MessageSimulator.o $(OBJDIR)/TestMessageProcessor1.o $(OBJDIR)/TestMessageProcessor2.o $(BAYES_PIPELINE_LIB) $(BAYES_CORE_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJDIR)/scenario_tests.o $(OBJDIR)/MessageSimulator.o $(OBJDIR)/TestMessageProcessor1.o $(OBJDIR)/TestMessageProcessor2.o \
	    -Wl,--whole-archive $(BAYES_PIPELINE_LIB) $(BAYES_CORE_LIB) \
	    -Wl,--no-whole-archive \
	    -ldl -lpthread
	@echo "Built $(TARGETSCENARIO)  →  run with:  ./$(TARGETSCENARIO)"

# Compile top-level entry point main.cpp into object file.
$(OBJDIR)/main.o: $(SRCDIR)/main.cpp $(SRCDIR)/main_app.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile deterministic scenario tests entry point.
$(OBJDIR)/scenario_tests.o: $(SRCDIR)/tests/scenario_tests.cpp $(SRCDIR)/messaging/MessageSimulator.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile MainApp implementation into object file.
$(OBJDIR)/main_app.o: $(SRCDIR)/main_app.cpp $(SRCDIR)/main_app.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile MessageSimulator implementation into object file.
$(OBJDIR)/MessageSimulator.o: $(SRCDIR)/messaging/MessageSimulator.cpp $(SRCDIR)/messaging/MessageSimulator.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile TestMessageProcessor1 implementation into object file.
$(OBJDIR)/TestMessageProcessor1.o: $(SRCDIR)/messaging/TestMessageProcessor1.cpp $(SRCDIR)/messaging/TestMessageProcessor1.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile TestMessageProcessor2 implementation into object file.
$(OBJDIR)/TestMessageProcessor2.o: $(SRCDIR)/messaging/TestMessageProcessor2.cpp $(SRCDIR)/messaging/TestMessageProcessor2.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Remove generated objects, primary build outputs, and dependency build dirs.
clean:
	rm -rf $(BUILDDIR) $(BAYES_PIPELINE_BUILD_DIR) obj libsource_lib.so main_app scenario_tests
