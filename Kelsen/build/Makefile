# Compiler and flags
CC = gcc
CFLAGS = -g -Wall
LDFLAGS = -lfl -ljansson

# Directories
SRC_DIR = ../src
BUILD_DIR = ../build
OUTPUT_DIR = ../output

# Source files and object files
SRCS = $(SRC_DIR)/ast.c $(SRC_DIR)/contract_generator.c $(SRC_DIR)/main.c $(SRC_DIR)/parser.tab.c $(SRC_DIR)/lex.yy.c $(SRC_DIR)/symbol_table.c
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# Target executable
TARGET = $(BUILD_DIR)/kelsen

# Rule to build everything
all: $(TARGET)

# Rule to link the target executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Rule to compile source files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)
	rm -rf $(OUTPUT_DIR)/*

.PHONY: all clean

