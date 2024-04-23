# Source files
SRC_DIR := src
TARGET := benchmark
PROFILES_DIR := profiles
NSYS_REPORT := $(PROFILES_DIR)/profile.nsys-rep
SRCS := $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/*.cu)
OMP_FLAGS := -Xcompiler -fopenmp

# Phony targets
.PHONY: build run clean profile-nsys view-report

# Build target
build: $(TARGET)
$(TARGET): $(SRCS)
	@echo "Building $(TARGET)"
	@nvcc -g -G $(OMP_FLAGS) $(SRCS) -o $@ -lgomp

# Rebuild the application
rebuild: 
	@echo "Cleaning $(TARGET)"
	@rm -f $(TARGET)
	@make -s build

# Build and run the application
run: build
	@echo "Running $(TARGET)"
	@./$(TARGET)

# Clean up generated files
clean:
	@echo "Cleaning up profiles..."
	@rm -rf $(PROFILES_DIR)/*

# Profile with Nsight Systems
profile-nsys: $(TARGET)
	@echo "Profiling $(TARGET) with Nsight Systems"
	@nsys profile --stats=true --output=$(NSYS_REPORT) ./$(TARGET)

# View the profiling report summary
view-report:
	@echo "Viewing Nsight Systems Report Summary"
	@nsys stats --report $(NSYS_REPORT)