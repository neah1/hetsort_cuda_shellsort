# Source files
SRC_DIR := src
OBJ_DIR := obj
SRCS := $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/*.cu)

# Compiler settings
CC=nvcc
CFLAGS=-g -G
LDFLAGS=
TARGET := $(OBJ_DIR)/shellsort

# Profiling settings
NSYS_PROFILE := nsys profile --stats=true --output=$(OBJ_DIR)/nsys_profile.qdrep
NCU_PROFILE := nv-nsight-cu-cli --metrics achieved_occupancy,sm_efficiency,achieved_active_warps_per_sm --export=$(OBJ_DIR)/ncu_profile.txt

# Phony targets
.PHONY: build rebuild run clean profile-nsys profile-ncu

# Build target
build: $(TARGET)
$(TARGET): $(SRCS)
	@echo "Building $(TARGET)"
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) $(SRCS) $(LDFLAGS) -o $@

# Run the application
run: rebuild
	@echo "Running $(TARGET)"
	@./$(TARGET)

# Clean up generated files
clean:
	@echo "Cleaning up..."
	@rm -rf $(OBJ_DIR)

# Rebuild the application
rebuild: clean build

# Profile with Nsight Systems
profile-nsys: $(TARGET)
	@echo "Profiling $(TARGET) with Nsight Systems"
	@$(NSYS_PROFILE) ./$(TARGET)

# Profile with Nsight Compute
profile-ncu: $(TARGET)
	@echo "Profiling $(TARGET) with Nsight Compute"
	@$(NCU_PROFILE) ./$(TARGET)