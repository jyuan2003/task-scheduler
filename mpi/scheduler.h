#include <limits>
#include <vector>
#include "../graph_representation/graph.h"
#include <utility>
struct Scheduler {
    // node -> node's color
    std::vector<int> colors;
    // color -> per_color runtime
    std::vector<int> runtime;
    int num_colors;
    // void simulate_compute(int weight);
    // void simulate_communicate(int weight);
};



void initialize(Scheduler &scheduler);
void bcast_compressed(CompressedGraph &compressed, int pid);
void bcast_raw(RawGraph &raw, int pid);

void all_to_all_sync_scheduler(Scheduler &scheduler, std::vector<int> &color_updates, std::vector<double> &runtime_updates,int pid); 

void report_program_stats(Scheduler &scheduler);
void update_scheduler(Scheduler &scheduler, std::vector<int> &color_updates, std::vector<int> &runtime_updates);

template <bool sharded, typename Graph>
void make_assignment(Graph &graph, Scheduler &scheduler, int pid, int num_iters, int batch_size) {
    std::vector<int> color_updates;
    std::vector<double> runtime_updates(scheduler.num_colors, 0.0);
    for (int iter = 0; iter < num_iters; iter ++) {
        if constexpr(sharded) {
            int global_id;
            for (const auto &shard : graph) {
                for (int local_id=0; local_id < shard.num_nodes(); local_id++) {
                    global_id = shard.local_to_global(local_id);
                    std::vector<int> global_id_neighbors = shard.get_node_out_neighbors(global_id);
                    double global_id_node_cost = shard.get_node_weight(global_id);
                    double global_id_cost = global_id_node_cost;
                    int global_id_color = scheduler.colors[global_id];
                    
                    double best_color_cost = std::numeric_limits<double>::infinity();
                    int best_color;

                    for (int new_color = 0; new_color < scheduler.num_colors; new_color ++) {
                        double global_id_new_cost = global_id_node_cost;
                        for (const auto & local_neighbor : global_id_neighbors) {
                            int global_neighbor = shard.local_to_global(local_neighbor);
                            global_id_new_cost += scheduler.colors[global_neighbor] == new_color ? 0 : shard.get_edge_weight(global_id, global_neighbor);
                        }
                        if (global_id_new_cost < best_color_cost) {
                            best_color_cost = global_id_new_cost;
                            best_color = new_color;
                        }
                        if (new_color == global_id_color) {
                            global_id_cost = global_id_new_cost;
                        }
                    }

                    if (best_color != global_id_color) {
                        color_updates.push_back(global_id);
                        color_updates.push_back(best_color);
                        runtime_updates[global_id_color] -= global_id_cost;
                        runtime_updates[best_color] += best_color_cost;
                    }
                }

                all_to_all_sync_scheduler(scheduler, color_updates, runtime_updates, pid);
                

            }
        }
        else {
            int pid_num_nodes = (graph.num_nodes() + scheduler.num_colors - 1) / scheduler.num_colors;
            for (int id = 0; id < pid_num_nodes; id += batch_size) {
                int true_batch_size = std::min(batch_size, pid_num_nodes - id);
                    
                for (int i = 0; i < true_batch_size; i++) {
                    int global_id = pid * pid_num_nodes + id + i;
                    std::vector<int> global_id_neighbors = graph.get_node_out_neighbors(global_id);
                    double global_id_node_cost = graph.get_node_weight(global_id);
                    double global_id_cost = global_id_node_cost;
                    int global_id_color = scheduler.colors[global_id];
                    
                    double best_color_cost = std::numeric_limits<double>::infinity();
                    int best_color;

                    for (int new_color = 0; new_color < scheduler.num_colors; new_color ++) {
                        double global_id_new_cost = global_id_node_cost;
                        for (const auto & global_neighbor : global_id_neighbors) {
                            global_id_new_cost += scheduler.colors[global_neighbor] == new_color ? 0 : graph.get_edge_weight(global_id, global_neighbor);
                        }
                        if (global_id_new_cost < best_color_cost) {
                            best_color_cost = global_id_new_cost;
                            best_color = new_color;
                        }
                        if (new_color == global_id_color) {
                            global_id_cost = global_id_new_cost;
                        }
                    }

                    if (best_color != global_id_color) {
                        color_updates.push_back(global_id);
                        color_updates.push_back(best_color);
                        runtime_updates[global_id_color] -= global_id_cost;
                        runtime_updates[best_color] += best_color_cost;
                    }
                }

                all_to_all_sync_scheduler(scheduler, color_updates, runtime_updates, pid);
                

            }
        }
    }
}

    
            