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
    int batch_size = 16;

    int opt;
    while ((opt = getopt(argc, argv, "f:i:b:")) != -1) {
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
    }

    bool sharded = false;
    bool compressed = false;

    // load input file
    std::filesystem::path input_path(input_graphname);

    std::string full_path = input_path.generic_string();


    if (full_path.find("/compressed/") != std::string::npos) {
        compressed = true;
    }
    else if (full_path.find("/compressed_sharded/") != std::string::npos) {
        sharded = true;
        compressed = true;
    }
    else if (full_path.find("/sharded/") != std::string::npos) {
        sharded = true;
    }
    int num_shards_per_process;
    Scheduler scheduler;
    scheduler.num_colors = num_processes;

    int num_nodes = 0;

    if (full_path.find("/small_") != std::string::npos) {
        num_nodes = 1000;
    }
    else if (full_path.find("/medium_") != std::string::npos) {
        num_nodes = 10000;
    }
    else if (full_path.find("/large_") != std::string::npos) {
        num_nodes = 100000;
    }

    scheduler.colors.resize(num_nodes);
    scheduler.runtime.resize(num_processes, 0.0);
    CompressedGraph compressed_graph;
    RawGraph raw_graph;
    std::vector<CompressedShardedGraph> compressed_sharded_graph;
    std::vector<ShardedGraph> sharded_graph;
    if (sharded) {
        if (num_processes <= 10) {
            num_shards_per_process = (10 + num_processes - 1) / num_processes;
        }

        else {
            num_shards_per_process = 1;
        }

        if (compressed) {
            int shard_start = pid * num_shards_per_process;
            int shard_end = std::min(10, shard_start + num_shards_per_process);

            for (int i = shard_start; i < shard_end; i++) {
                CompressedShardedGraph g;
                const std::string fname = input_graphname + "/" + "compressed_sharded" + "_" + std::to_string(i) + ".txt";
                load_compressed_sharded(g, fname);
                compressed_sharded_graph.push_back(g);
            }
        }

        else {
            int shard_start = pid * num_shards_per_process;
            int shard_end = std::min(10, shard_start + num_shards_per_process);
            for (int i = shard_start; i < shard_end; i++) {
                ShardedGraph g;
                const std::string fname = input_graphname + "/" +  "sharded" + "_" + std::to_string(i) + ".txt";
                load_sharded(g, fname);
                sharded_graph.push_back(g);
            }
        }
    }
    else {
        if (pid == 0) {
            if (compressed) {
                std::filesystem::path compressed_path = input_path;
                if (std::filesystem::is_directory(compressed_path)) {
                    compressed_path /= "compressed.txt";
                }
                load_compressed(compressed_graph, compressed_path.generic_string());
            }
            else {   
                load_raw(raw_graph, input_graphname);
            }
        }
    }
    const auto compute_start = std::chrono::steady_clock::now();
    if (pid == 0) {
        initialize(scheduler);
    }

    MPI_Bcast(scheduler.colors.data(), scheduler.colors.size(), MPI_INT, 0, MPI_COMM_WORLD);
    
    if (sharded) {
        if (compressed) {
            make_assignment<true>(compressed_sharded_graph, scheduler, pid, num_iters, batch_size);
        }
        else {
            make_assignment<true>(sharded_graph, scheduler, pid, num_iters, batch_size);
        }
    }
    else {
        if (compressed) {
            bcast_compressed(compressed_graph, pid);
            make_assignment<false>(compressed_graph, scheduler, pid, num_iters, batch_size);
        }
        else {
            bcast_raw(raw_graph, pid);
            make_assignment<false>(raw_graph, scheduler, pid, num_iters, batch_size);
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
