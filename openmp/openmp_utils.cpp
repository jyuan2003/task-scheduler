#include "scheduler.h"
#include <algorithm>
#include <omp.h>
#include <random>


void initialize(Scheduler &scheduler) {
    std::random_device rd;
    std::mt19937 gen(rd());  
    std::uniform_int_distribution<int> dist(0, scheduler.num_colors - 1);
    for (std::size_t i = 0; i < scheduler.colors.size(); i++) {
        scheduler.colors[i] = dist(gen);
    }
}

void update_scheduler(Scheduler &scheduler, int old_color, int new_color, int node){
    if (old_color == new_color) return;
    double old_in = 0.0;
    double old_out = 0.0;
    double new_in = 0.0;
    double new_out = 0.0;
    double all_out = 0.0;
    for (const auto &out : out_neighbors){
        if (scheduler.colors[out] == old_color){
            old_out += graph.get_edge_weight(v, out);
        }
        else{
            all_out += graph.get_edge_weight(v, out);
        if (scheduler.colors[out] == new_color)
            new_out += graph.get_edge_weight(v, out);                  
        }
    }
    for (const auto &in : in_neighbors){
        if (scheduler.colors[in] == new_color){
            new_in += graph.get_edge_weight(in, v);
        }
        else if (scheduler.colors[in] == old_color){
            old_in += graph.get_edge_weight(in, v);
        }
    }
    scheduler.runtime[old_color] -= graph.get_node_weight(v);
    scheduler.runtime[old_color] -= all_out;
    scheduler.runtime[old_color] += old_in;
    scheduler.runtime[new_color] += graph.get_node_weight(v);
    scheduler.runtime[new_color] -= new_in;
    scheduler.runtime[new_color] += old_out + all_out - new_out;
    scheduler.max_runtime = *std::max_element(scheduler.runtime.begin(), scheduler.runtime.end());
}

void report_program_stats(Scheduler &scheduler) {
    double span = scheduler.max_runtime;
    std::cout << "Runtime span:" << " " << span << std::endl;
}