#include "scheduler.h"
#include <algorithm>
#include <mpi.h>
#include <random>
#include <chrono>
#include <cmath>

void initialize_random(Scheduler &scheduler) {
    std::random_device rd;
    std::mt19937 gen(rd());  
    std::uniform_int_distribution<int> dist(0, scheduler.num_colors - 1);
    for (std::size_t i = 0; i < scheduler.colors.size(); i++) {
        scheduler.colors[i] = dist(gen);
    }
}

void initialize_contiguous(Scheduler &scheduler) {
    for (std::size_t i = 0; i < scheduler.colors.size(); i++) {
        int color = (int)(i * scheduler.num_colors / scheduler.colors.size());
        scheduler.colors[i] = std::min(color, scheduler.num_colors - 1);
    }
}



void bcast_compressed(CompressedGraph &compressed, int pid) {
    int num_nodes = 0;

    if (pid == 0) {
        num_nodes = compressed.nodes.size();
    }
        
    MPI_Bcast(&num_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (pid != 0) {
        compressed.nodes.resize(num_nodes);
    }    

    for (int i = 0; i < num_nodes; i++) {

        int id = 0;
        int num_groups = 0;
        double weight = 0.0;

        if (pid == 0) {
            id = compressed.nodes[i].id;
            num_groups = compressed.nodes[i].groups.size();
            weight = compressed.nodes[i].weight;
        }

        MPI_Bcast(&id, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&weight, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Bcast(&num_groups, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (pid != 0) {
            compressed.nodes[i].id = id;
            compressed.nodes[i].weight = weight;
            compressed.nodes[i].groups.resize(num_groups);
        }

        for (int j = 0; j < num_groups; j++) {
            int bytes = 0;
            int num_edges = 0;
            int num_out_neighbors = 0;
            int num_weights = 0;

            if (pid == 0) {
                bytes = compressed.nodes[i].groups[j].header.bytes;
                num_edges = compressed.nodes[i].groups[j].header.num_edges;
                num_out_neighbors = compressed.nodes[i].groups[j].out_neighbors.size();
                num_weights = compressed.nodes[i].groups[j].weights.size();
            }

            MPI_Bcast(&bytes, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD); 
            
            MPI_Bcast(&num_out_neighbors, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&num_weights, 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (pid != 0) {
                compressed.nodes[i].groups[j].header.bytes = bytes;
                compressed.nodes[i].groups[j].header.num_edges = num_edges;
                compressed.nodes[i].groups[j].out_neighbors.resize(num_out_neighbors);
                compressed.nodes[i].groups[j].weights.resize(num_weights);
            }

            if (num_out_neighbors > 0) {
                MPI_Bcast(
                    compressed.nodes[i].groups[j].out_neighbors.data(),
                    num_out_neighbors,
                    MPI_INT,
                    0,
                    MPI_COMM_WORLD
                );
            }

            if (num_weights > 0) {
                MPI_Bcast(
                    compressed.nodes[i].groups[j].weights.data(),
                    num_weights,
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD
                );
            }

        }
    }
}


void bcast_raw(RawGraph &raw, int pid) {
    int num_nodes = 0;
    int num_edges = 0;

    if (pid == 0) {
        num_nodes = raw.nodes.size();
        num_edges = raw.edges.size();
    }

    MPI_Bcast(&num_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (pid != 0) {
        raw.nodes.resize(num_nodes);

        raw.edges.resize(num_edges);
    }

    for (int i = 0; i < num_nodes; i++) {
        int id;
        double weight;

        if (pid == 0) {
            id = raw.nodes[i].id;
            weight = raw.nodes[i].weight;

        }

        MPI_Bcast(&id, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&weight, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (pid != 0) {
            raw.nodes[i].id = id;
            raw.nodes[i].weight = weight;
        }

    }

    for (int i = 0; i < num_edges; i++) {
        int u, v;
        double weight;

        if (pid == 0) {
            u = raw.edges[i].u;
            v = raw.edges[i].v;
    
            weight = raw.edges[i].weight;
        }

        MPI_Bcast(&u, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&v, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&weight, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (pid != 0) {
            raw.edges[i].u = u;
            raw.edges[i].v = v;
            raw.edges[i].weight = weight;
        }
    }

}





void report_program_stats(Scheduler &scheduler) {
    double span = *std::max_element(scheduler.runtime.begin(), scheduler.runtime.end());
    std::cout << "Runtime span:" << " " << span << std::endl;
}

