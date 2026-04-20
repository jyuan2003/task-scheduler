#include "scheduler.h"
#include <algorithm>
#include <mpi.h>
#include <random>

void initialize(Scheduler &scheduler) {
    std::random_device rd;
    std::mt19937 gen(rd());  
    std::uniform_int_distribution<int> dist(0, scheduler.num_colors - 1);
    for (std::size_t i = 0; i < scheduler.colors.size(); i++) {
        scheduler.colors[i] = dist(gen);
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

void update_scheduler(Scheduler &scheduler, std::vector<int> &color_updates, std::vector<double> &runtime_updates) {
    for (int i = 0; i < color_updates.size(); i += 2) {
        scheduler.colors[color_updates[i]] = color_updates[i + 1];
    }
    for (int i = 0; i < runtime_updates.size(); i++) {
        scheduler.runtime[i] += runtime_updates[i];
    }
}

void all_to_all_sync_scheduler(
    Scheduler &scheduler, 
    std::vector<int> &color_updates, 
    std::vector<double> &runtime_updates,
    int pid
) {
    
    int num_processes = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    int num_local_color_updates = color_updates.size();
    std::vector<int> num_global_color_updates(num_processes);

    MPI_Allgather(&num_local_color_updates, 1, MPI_INT, num_global_color_updates.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> global_color_updates;
    std::vector<double> global_runtime_updates(scheduler.num_colors, 0.0);
    std::vector<int> global_color_offsets(num_processes, 0);

    int total_global_color_updates = 0;

    for (int i = 0; i < num_processes; i++) {
        if (i > 0) {
            global_color_offsets[i] = global_color_offsets[i - 1] + num_global_color_updates[i - 1];
        }
        total_global_color_updates += num_global_color_updates[i];
    }

    global_color_updates.resize(total_global_color_updates);

    if (total_global_color_updates > 0) {
        MPI_Allgatherv(
            color_updates.data(), 
            num_local_color_updates, 
            MPI_INT,
            global_color_updates.data(),
            num_global_color_updates.data(),
            global_color_offsets.data(),
            MPI_INT,
            MPI_COMM_WORLD
        );
    }



    MPI_Allreduce(
        runtime_updates.data(),
        global_runtime_updates.data(),
        scheduler.num_colors,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD
    ); 

    update_scheduler(scheduler, global_color_updates, global_runtime_updates);
    color_updates.clear();
    std::fill(runtime_updates.begin(), runtime_updates.end(), 0.0);
    
}




void report_program_stats(Scheduler &scheduler) {
    double span = *std::max_element(scheduler.runtime.begin(), scheduler.runtime.end());
    std::cout << "Runtime span:" << " " << span << std::endl;
}



