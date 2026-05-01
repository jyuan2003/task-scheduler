CXX ?= g++
CXXFLAGS ?= -std=c++20 -Wall -Wextra -pedantic

TARGET := graph_main
SRCS := graph_main.cpp graph_utils.cpp
HDRS := graph.h

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)


run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) 
