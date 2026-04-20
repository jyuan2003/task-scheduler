MPICXX ?= mpic++
MPICXXFLAGS ?= -std=c++20 -w

TARGET := mpi_main
SRCS := mpi_main.cpp scheduler_utils.cpp ../graph_representation/graph_utils.cpp
HDRS := scheduler.h ../graph_representation/graph.h

.PHONY: all clean 

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(MPICXX) $(MPICXXFLAGS) $(SRCS) -o $(TARGET)


clean:
	rm -f $(TARGET)
