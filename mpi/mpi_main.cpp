#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <mpi.h>
#include <unistd.h>
#include <filesystem>
#include "scheduler.h"



int main(int argc, char *argv[]) {

    // Adapted from asst4 codebase
    const auto init_start = std::chrono::steady_clock::now();

    int pid;
    int num_processes;


    MPI_Init(&argc, &argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
      
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);


    std::string input_graphname;

    int num_iters = 5;
    int batch_size = 128;
    std::string init_mode = "random";
    std::string rep;
    int opt;
    while ((opt = getopt(argc, argv, "f:i:b:m:r:")) != -1) {
        switch (opt) {
        case 'f':
            input_graphname = optarg;
            break;
        case 'i':
            num_iters = atoi(optarg);
            break;
        case 'b':
            batch_size = atoi(optarg);
            break;
        case 'm':
            init_mode = optarg;
            break;
        case 'r':
            rep = optarg;
            break;
        default:
            if (pid == 0) {
                std::cerr << "Usage: " << argv[0] << " -f input_graphname [-i num_iters] -b batch_size\n";
            }

            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    if (input_graphname.empty() || num_iters <= 0 || batch_size <= 0) {
        if (pid == 0) {
            std::cerr << "Usage: " << argv[0] << " -f input_graphname [-i num_iters] -b batch_size\n";
        }

        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    if (pid == 0) {
        std::cout << "Number of processes: " << num_processes << '\n';
        std::cout << "Simulated iterations: " << num_iters << '\n';
        std::cout << "Input file: " << input_graphname << '\n';
        std::cout << "Batch size: " << batch_size << '\n';
        std::cout << "Initialization: " << init_mode << '\n';
    }



    bool sharded = false;
    bool compressed = false;

    // load input file
    std::filesystem::path input_path(input_graphname);

    std::string full_path = input_path.generic_string();


    int num_shards_per_process;
    Scheduler scheduler;
    scheduler.num_colors = num_processes;
    
    CompressedGraph compressed_graph;
    RawGraph raw_graph;
    std::vector<CompressedShardedGraph> compressed_sharded_graph;
    std::vector<ShardedGraph> sharded_graph;

    int num_shards = num_processes;
    load_raw(raw_graph, input_graphname);
    
    int num_nodes = raw_graph.num_nodes();
    
    scheduler.colors.resize(num_nodes);
    scheduler.runtime.resize(num_processes, 0.0);
    
    if (rep == "compressed") {
        raw_to_compressed(raw_graph, compressed_graph);
        compressed = true;
    }
    else if (rep == "sharded") {
        raw_to_sharded(raw_graph, sharded_graph, num_shards);
        sharded = true;
    }
    else if (rep == "compressed_sharded") {
        compressed = true;
        sharded = true;
        raw_to_sharded(raw_graph, sharded_graph, num_shards);
        for (auto shard : sharded_graph) {
            CompressedShardedGraph compressed_sharded;
            sharded_to_compressedsharded(shard, compressed_sharded);
            compressed_sharded_graph.push_back(compressed_sharded);
        }
    }

    const auto compute_start = std::chrono::steady_clock::now();
    
    if (pid == 0) {
        if (init_mode == "random") {
            initialize_random(scheduler);
        }
        else {
            initialize_contiguous(scheduler);
        }
        
    }

    MPI_Bcast(scheduler.colors.data(), scheduler.colors.size(), MPI_INT, 0, MPI_COMM_WORLD);
    
    if (sharded) {
        if (compressed) {
            do_coloring<true>(compressed_sharded_graph, scheduler, pid, num_iters, batch_size);
        }
        else {
            do_coloring<true>(sharded_graph, scheduler, pid, num_iters, batch_size);
        }
    }
    else {
        if (compressed) {
            bcast_compressed(compressed_graph, pid);
            do_coloring<false>(compressed_graph, scheduler, pid, num_iters, batch_size);
        }
        else {
            bcast_raw(raw_graph, pid);
            do_coloring<false>(raw_graph, scheduler, pid, num_iters, batch_size);
        }
    }


    if (pid == 0) {
        const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
        std::cout << "Computation time (sec): " << std::fixed << std::setprecision(10) << compute_time << '\n';

        report_program_stats(scheduler);
    }

    
    MPI_Finalize();
    return 0;
}
