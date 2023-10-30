# Makefile for CUDA applications

# Source files
SRC=shellsort.cu

# Compiler settings
CC=nvcc
CFLAGS=-g -G
LDFLAGS=
TARGET=shellsort
OBJ=$(SRC:.cu=.o)

# Build the application
build:
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(TARGET) $(SRC)

# Rebuild the application
rebuild: clean build

# Run the application
run: rebuild
	./$(TARGET)

# Clean up generated files
clean:
	rm -f $(TARGET)

.PHONY: build rebuild run clean
