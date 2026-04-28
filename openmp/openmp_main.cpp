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
#include <unistd.h>
#include <filesystem>
#include "openmp_scheduler.h"

int main(int argc, char *argv[]) {
    const auto init_start = std::chrono::steady_clock::now();
    int num_processes;
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
            std::cerr << "Usage: " << argv[0] << " -f input_graphname [-i num_iters] -b batch_size\n";
            exit(EXIT_FAILURE);
        }
    }

    if (input_graphname.empty() || num_iters <= 0 || batch_size <= 0) {
        std::cerr << "Usage: " << argv[0] << " -f input_graphname [-i num_iters] -b batch_size\n";
        exit(EXIT_FAILURE);
    }


    std::cout << "Number of processes: " << num_processes << '\n';
    std::cout << "Simulated iterations: " << num_iters << '\n';
    std::cout << "Input file: " << input_graphname << '\n';
    std::cout << "Batch size: " << batch_size << '\n';
    bool compressed;
    if (full_path.find("/compressed/") != std::string::npos 
    || full_path.find("/compressed_sharded/") != std::string::npos) {
        compressed = true;
    }
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
    std::vector<std::vector<int>> nodes(num_processes);
    std::vector<int> starting_batch(num_processes);
    std::vector<int> starting_node(num_processes);
    initialize(scheduler);
    starting_batch[0] = 0;
    starting_node[0] = 0;
    if (compressed){
        int available_node = 0;
        int total_batches = 0;
        int total_degree = 2 * compressed_graph.num_edges();
        for (int i = 0; i < num_processes; i++){
            int process_total_degree = 0;
            while (process_total_degree < total_degree / num_processes
                && available_node < compressed_graph.num_nodes()){
                    nodes[i].push_back(available_node);
                    process_total_degree += compressed_graph.get_node_out_neighbors(available_node).size();
                    available_node ++;
            }
            if (i < num_processes - 1){
                starting_batch[i + 1] = starting_batch[i] + (nodes[i].size() + batch_size - 1) / batch_size;
            }
        }
    }
    else{
        int available_node = 0;
        int total_degree = 2 * raw_graph.num_edges();
        for (int i = 0; i < num_processes; i++){
            int process_total_degree = 0;
            while (process_total_degree < total_degree / num_processes
                && available_node < raw_graph.num_nodes()){
                nodes[i].push_back(available_node);
                process_total_degree += raw_graph.get_node_out_neighbors(available_node).size();
                available_node ++;
            }
            if (i < num_processes - 1){
                starting_batch[i + 1] = starting_batch[i] + (nodes[i].size() + batch_size - 1) / batch_size;
            }
        }
    }
    int total_batches = starting_batch[-1] + (graph.num_nodes() - starting_node[-1] + batch_size - 1) / batch_size;
    std::vector<omp_lock_t> locks(total_batches);
    std::vector<bool> batch_done(total_batches);
    for (auto &lk : locks) {
        omp_init_lock(&lk);
    }
    const auto compute_start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < num_iters; ++iter){
        batch_done.assign(total_batches, false);
        #pragma omp parallel num_threads(num_processes){
            int pid = omp_get_thread_num();
            if (compressed){
                make_assignment(&compressed_graph, &scheduler, pid, batch_size, &starting_node, &locks, &starting_batch, &batch_done);
            }
            else{
                make_assignment(&raw_graph, &scheduler, pid, batch_size, &starting_node, &locks, &starting_batch, &batch_done);
            }
        }
    }
}