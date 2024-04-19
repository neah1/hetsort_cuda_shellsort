# Source files
SRC_DIR := src
BUILD_DIR := build
TARGET := $(BUILD_DIR)/main
NSYS_REPORT := $(BUILD_DIR)/profile.nsys-rep
SRCS := $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/*.cu)
OMP_FLAGS := -Xcompiler -fopenmp --expt-relaxed-constexpr

# Phony targets
.PHONY: build rebuild run rerun clean profile-nsys view-report

# Build target
build: $(TARGET)
$(TARGET): $(SRCS)
	@echo "Building $(TARGET)"
	@mkdir -p $(BUILD_DIR)
	@nvcc -g -G $(OMP_FLAGS) $(SRCS) -o $@ -lgomp

# Rebuild the application
rebuild: 
	@echo "Cleaning $(TARGET)"
	@rm -f $(TARGET)
	@make -s build

# Run the application
run:
	@./$(TARGET)

# Clean up generated files
clean:
	@echo "Cleaning up..."
	@rm -rf $(BUILD_DIR)

# Profile with Nsight Systems
profile-nsys: $(TARGET)
	@echo "Profiling $(TARGET) with Nsight Systems"
	@nsys profile --stats=true --output=$(NSYS_REPORT) ./$(TARGET)

# View the profiling report summary
view-report:
	@echo "Viewing Nsight Systems Report Summary"
	@nsys stats --report $(NSYS_REPORT)