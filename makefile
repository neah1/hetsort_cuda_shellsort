# Makefile for CUDA applications

# Compiler settings
CC=nvcc
CFLAGS=-g -G
LDFLAGS=
TARGET=shellsort

# Source and Object files
SRC=shellsort_basic.cu
OBJ=$(SRC:.cu=.o)

# Build the application
build: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

# Run the application
run: build
	./$(TARGET)

# Debug the application (this will launch cuda-gdb)
debug: build
	cuda-gdb --args ./$(TARGET)

# Clean up generated files
clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: build run debug clean
