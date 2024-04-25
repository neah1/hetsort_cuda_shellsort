# Source files
TARGET := main
SRC_DIR := src
PROFILES_DIR := profiles
SRCS := $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/*.cu)

# Phony targets
.PHONY: build rebuild run clean clean-sqlite

# Build target
build: $(TARGET)
$(TARGET): $(SRCS)
	@echo "Building $(TARGET)"
	@nvcc -g -G -Xcompiler -fopenmp $(SRCS) -o $@ -lgomp

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

# Clean up sqlite files
clean-sqlite:
	@echo "Cleaning up sqlite files..."
	@rm -rf $(PROFILES_DIR)/*.sqlite
