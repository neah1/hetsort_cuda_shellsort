# Makefile for CUDA applications

# Compiler settings
CC=nvcc
CFLAGS=-g -G
LDFLAGS=
TARGET=shellsort

# Source and Object files
SRC=shellsort.cu
OBJ=$(SRC:.cu=.o)

# Build the application
build: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

# Rebuild the application
rebuild: clean build

# Run the application
run: build
	./$(TARGET)

# Clean up generated files
clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: build rebuild run clean
