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
#include <omp.h>
#include "openmp_scheduler.h"
#include <unistd.h>
#include <filesystem>

int main(int argc, char *argv[]) {
    const auto init_start = std::chrono::steady_clock::now();
    int num_processes = 1;
    std::string input_graphname;
    int num_iters = 5;
    int batch_size = 16;
    int color = 1;
    int opt;
    std::string rep;
    while ((opt = getopt(argc, argv, "f:i:n:b:c:r:")) != -1) {
        switch (opt) {
        case 'n':
            num_processes = atoi(optarg);
            break;
        case 'f':
            input_graphname = optarg;
            break;
        case 'i':
            num_iters = atoi(optarg);
            break;
        case 'b':
            batch_size = atoi(optarg);
            break;
        case 'c':
            color = atoi(optarg);
            break;
        case 'r':
            rep = optarg;
            break;
        default:
            std::cerr << "Usage: " << argv[0] << " -f input_graphname [-i num_iters] -n num_processes -b batch_size -c num_colors -r repository\n";
            exit(EXIT_FAILURE);
        }
    }

    if (input_graphname.empty() || num_iters <= 0 || batch_size <= 0) {
        std::cerr << "Usage: " << argv[0] << " -f input_graphname [-i num_iters] -n num_processes -b batch_size -c num_colors\n";
        exit(EXIT_FAILURE);
    }


    std::cout << "Number of processes: " << num_processes << '\n';
    std::cout << "Number of colors: " << color << '\n';
    std::cout << "Simulated iterations: " << num_iters << '\n';
    std::cout << "Input file: " << input_graphname << '\n';
    std::cout << "Batch size: " << batch_size << '\n';
    std::cout << "Repository: " << rep << '\n';
    bool compressed = false;
    bool shard = false;
    std::filesystem::path input_path(input_graphname);

    std::string full_path = input_path.generic_string();

    if (full_path.find("/compressed/") != std::string::npos 
    || full_path.find("/compressed_sharded/") != std::string::npos) {
        compressed = true;
        if (full_path.find("/compressed_sharded/") != std::string::npos)
            shard = true;
    }
    Scheduler scheduler;
    scheduler.num_colors = color;

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
    if (rep == "compressed") {
        raw_to_compressed(raw_graph, compressed_graph);
        compressed = true;
    }
    std::vector<std::vector<int>> nodes(num_processes);
    std::vector<int> starting_node(num_processes);
    initialize(scheduler);
    if (compressed){
        num_nodes = compressed_graph.num_nodes();
    }
    else{
        num_nodes = raw_graph.num_nodes();
    }
    int total_batches = (num_nodes + batch_size - 1) / batch_size;
    int batch_per_process = (total_batches + num_processes - 1) / num_processes;
    std::vector<int> starting_batch(num_processes);
    for (int i = 0; i < num_processes; i++){
        starting_batch[i] = i * batch_per_process;
        starting_node[i] = i * batch_per_process * batch_size;
    }
    std::vector<omp_lock_t> locks(total_batches);
    std::vector<bool> batch_done(total_batches);
    for (auto &lk : locks) {
        omp_init_lock(&lk);
    }
    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    const auto compute_start = std::chrono::steady_clock::now();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';
    for (int iter = 0; iter < num_iters; ++iter){
        batch_done.assign(total_batches, false);
        #pragma omp parallel num_threads(num_processes)
        {
            int pid = omp_get_thread_num();
            if (compressed){
                make_assignment(compressed_graph, scheduler, pid, batch_size, starting_node, locks, starting_batch, batch_done);
            }
            else{
                make_assignment(raw_graph, scheduler, pid, batch_size, starting_node, locks, starting_batch, batch_done);
            }
        }
    }
    for (auto &lk : locks){
        omp_destroy_lock(&lk);
    }
    omp_destroy_lock(&(scheduler.lock));
    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " << std::fixed << std::setprecision(10) << compute_time << '\n';
    return 0;
}