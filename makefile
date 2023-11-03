# Makefile for CUDA applications

# Source files
SRC_DIR := src
BIN_DIR := obj
SRCS := $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/*.cu)

# Compiler settings
CC=nvcc
CFLAGS=-g -G
LDFLAGS=
TARGET := $(BIN_DIR)/shellsort

# Phony targets
.PHONY: build rebuild run clean

# Default build target
build: $(TARGET)
$(TARGET): $(SRCS)
	@echo "Building $(TARGET)"
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(SRCS) $(LDFLAGS) -o $@

# Run the application
run: rebuild
	@echo "Running $(TARGET)"
	@./$(TARGET)

# Clean up generated files
clean:
	@echo "Cleaning up..."
	@rm -rf $(BIN_DIR)

# Rebuild the application
rebuild: clean build