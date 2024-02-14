# Source files
SRC_DIR := src
BUILD_DIR := build
TARGET := $(BUILD_DIR)/shellsort
NSYS_REPORT := $(BUILD_DIR)/profile.nsys-rep
SRCS := $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/*.cu)

# Phony targets
.PHONY: build rebuild run clean profile-nsys view-report

# Build target
build: $(TARGET)
$(TARGET): $(SRCS)
	@echo "Building $(TARGET)"
	@mkdir -p $(BUILD_DIR)
	@nvcc -g -G $(SRCS) -o $@

# Rebuild the application
rebuild: 
	@echo "Cleaning $(TARGET)"
	@rm -f $(TARGET)
	@make -s build

# Run the application
run: rebuild
	@echo "Running $(TARGET)"
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