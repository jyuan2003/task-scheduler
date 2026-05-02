#include <limits>
#include <vector>
#include "../graph_representation/graph.h"
#include <omp.h>
#include <utility>
#include <algorithm>
struct Scheduler {
    // node -> node's color
    std::vector<int> colors;
    // color -> per_color runtime
    std::vector<double> runtime;
    int num_colors;
    int max_runtime;
    omp_lock_t lock;
};

void initialize(Scheduler &scheduler);
void report_program_stats(Scheduler &scheduler);
template <typename Graph>
void update_scheduler(Scheduler &scheduler, Graph &graph, std::vector<int> &color_updates, int batch_node, int max_node){
    omp_set_lock(&(scheduler.lock));
    for (int v = batch_node; v < max_node; v++){
    double old_in = 0.0;
    double old_out = 0.0;
    double new_in = 0.0;
    double new_out = 0.0;
    double all_out = 0.0;
    int old_color = scheduler.colors[v];
    int new_color = color_updates[v - batch_node];
    std::vector<int> out_neighbors = graph.get_node_out_neighbors(v);
    std::vector<int> in_neighbors = graph.get_node_in_neighbors(v);
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
    omp_unset_lock(&(scheduler.lock));
}

template <typename Graph>
void assign_node(Graph &graph, Scheduler &scheduler, int v, std::vector<int> &color_updates, int batch_node){
    std::vector<int> out_neighbors = graph.get_node_out_neighbors(v);
            std::vector<int> in_neighbors = graph.get_node_in_neighbors(v);
            double best_cost = std::numeric_limits<double>::infinity();
            int old_color = scheduler.colors[v];
            int best_color;
            for (int new_color = 0; new_color < scheduler.num_colors; ++new_color){
                if (new_color != old_color){
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
                    std::vector<double> copy_scheduler;
                    copy_scheduler = scheduler.runtime;
                    copy_scheduler[old_color] -= graph.get_node_weight(v);
                    copy_scheduler[old_color] -= all_out;
                    copy_scheduler[old_color] += old_in;
                    copy_scheduler[new_color] += graph.get_node_weight(v);
                    copy_scheduler[new_color] -= new_in;
                    copy_scheduler[new_color] += old_out + all_out - new_out;
                    double temp = *std::max_element(copy_scheduler.begin(), copy_scheduler.end());
                    if (best_cost > temp){
                        best_cost = temp;
                        best_color = new_color;
                    }
                }
                else{
                    if (best_cost > *std::max_element(scheduler.runtime.begin(), scheduler.runtime.end())){
                        best_cost = *std::max_element(scheduler.runtime.begin(), scheduler.runtime.end());
                        best_color = old_color;
                    }
                }
            }
            color_updates[v - batch_node] = best_color;
}
template <typename Graph>
void make_assignment(Graph &graph, Scheduler &scheduler, int pid, int batch_size,
    std::vector<int> &starting_node, std::vector<omp_lock_t> &locks, std::vector<int> &starting_batch, std::vector<bool> &batch_done){
    int batch_node = starting_node[pid];
    int max_batch = (pid == starting_batch.size() - 1) ? batch_done.size() : starting_batch[pid+1];
    int max_node = (pid == starting_node.size() - 1) ? graph.num_nodes(): starting_node[pid+1];
    std::vector<int> color_updates(batch_size);
    for (int batch = starting_batch[pid]; batch < max_batch; ++batch){
        omp_set_lock(&locks[batch]);
        if (batch_done[batch]){
            omp_unset_lock(&locks[batch]);
            batch_node = std::min(batch_node + batch_size, max_node);
            continue;
        }
        batch_done[batch] = true;
        omp_unset_lock(&locks[batch]);
        for (int v = batch_node; v < std::min(batch_node + batch_size, max_node); ++v){
            assign_node(graph, scheduler, v, color_updates, batch_node);
        }
        update_scheduler(scheduler, graph, color_updates, batch_node, std::min(batch_node + batch_size, max_node));
        batch_node = std::min(batch_node + batch_size, max_node);
    }
    for (int proc = pid+1; proc < starting_batch.size(); ++proc){
        int max_batch = (proc == starting_batch.size() - 1) ? batch_done.size() : starting_batch[proc+1];
        int min_batch = starting_batch[proc];
        int batch_node = starting_node[proc];
        int max_node = (proc == starting_node.size() - 1) ? graph.num_nodes(): starting_node[proc+1];
        for (int batch = min_batch; batch < max_batch; ++batch){
            omp_set_lock(&locks[batch]);
            if (batch_done[batch]){
                omp_unset_lock(&locks[batch]);
                batch_node = std::min(batch_node + batch_size, max_node);
                continue;
            }
            batch_done[batch] = true;
            omp_unset_lock(&locks[batch]);
            for (int v = batch_node; v < std::min(batch_node + batch_size, graph.num_nodes()); ++v){
                assign_node(graph, scheduler, v, color_updates, batch_node);
            }
            update_scheduler(scheduler, graph, color_updates, batch_node, std::min(batch_node + batch_size, max_node));
            batch_node = std::min(batch_node + batch_size, max_node);
        }
    }
    for (int proc = pid-1; proc >= 0; --proc){
        int max_batch = (proc == starting_batch.size() - 1) ? batch_done.size() : starting_batch[proc+1];
        int min_batch = starting_batch[proc];
        int batch_node = starting_node[proc];
        int max_node = (proc == starting_node.size() - 1) ? graph.num_nodes(): starting_node[proc+1];
        for (int batch = min_batch; batch < max_batch; ++batch){
            omp_set_lock(&locks[batch]);
            if (batch_done[batch]){
                omp_unset_lock(&locks[batch]);
                batch_node = std::min(batch_node + batch_size, max_node);
                continue;
            }
            batch_done[batch] = true;
            omp_unset_lock(&locks[batch]);
            for (int v = batch_node; v < std::min(batch_node + batch_size, graph.num_nodes()); ++v){
                assign_node(graph, scheduler, v, color_updates, batch_node);
            }
            update_scheduler(scheduler, graph, color_updates, batch_node, std::min(batch_node + batch_size, max_node));
            batch_node = std::min(batch_node + batch_size, max_node);
        }
    }
}