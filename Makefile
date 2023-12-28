# Compilador
CXX := g++

# Opciones de compilación
CXXFLAGS := -std=c++17 -Wpedantic -I./include -fsanitize=address

# Directorios
SRC_DIR := ./src
INC_DIR := ./include
LIB_DIR := ./lib
BUILD_DIR := ./build

# Archivos fuente
SRCS := $(wildcard $(SRC_DIR)/*.cc $(LIB_DIR)/*.cc)

# Objetos generados
OBJS := $(patsubst $(SRC_DIR)/%.cc,$(BUILD_DIR)/%.o,$(SRCS))

# Nombre del ejecutable
TARGET := programa.out

# Regla principal
all: $(BUILD_DIR) $(TARGET)

# Regla para el ejecutable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Regla para los objetos
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Regla para crear el directorio de construcción
$(BUILD_DIR):
	mkdir -p $@

# Limpiar archivos generados
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

# No realizar ninguna acción para los targets "clean" y "all"
.PHONY: clean all