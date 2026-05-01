#include <limits>
#include <vector>
#include "../graph_representation/graph.h"
#include <utility>
#include <algorithm>
#include <cmath>
#include <mpi.h>

struct Scheduler {
    // node -> node's color
    std::vector<int> colors;
    // proc/color -> proc runtime
    std::vector<double> runtime;
    int num_colors;

};


void initialize_random(Scheduler &scheduler);
void initialize_contiguous(Scheduler &scheduler);

void bcast_compressed(CompressedGraph &compressed, int pid);
void bcast_raw(RawGraph &raw, int pid);

template <bool sharded, typename Graph>
void update_scheduler_runtime(Graph &graph, Scheduler &scheduler) {
    if constexpr(sharded) {
        for (const auto &g : graph) {
            for (int i = g.node_start; i < g.node_end; i++) {
                int color = scheduler.colors[i];
                scheduler.runtime[color] += g.get_node_weight(i);
                std::vector<int> outs = g.get_node_out_neighbors(i);
                for (const auto &out : outs) {
                    int out_color = scheduler.colors[out];
                    scheduler.runtime[color] += color == out_color ? 0 : g.get_edge_weight(i, out);
                }
            }
        }
    } else {
        for (int i = 0; i < scheduler.colors.size(); i++) {
            int color = scheduler.colors[i];
            scheduler.runtime[color] += graph.get_node_weight(i);
            std::vector<int> outs = graph.get_node_out_neighbors(i);
            for (const auto &out : outs) {
                int out_color = scheduler.colors[out];
                scheduler.runtime[color] += color == out_color ? 0 : graph.get_edge_weight(i, out);
            }
        }
    }
    
}



void report_program_stats(Scheduler &scheduler);

template <bool sharded, typename Graph>
void do_coloring(Graph &graph, Scheduler &scheduler, int pid, int num_iters, int batch_size) {
    std::vector<int> color_updates;
    update_scheduler_runtime<sharded>(graph, scheduler);
    // adapted from assignment 4 codebase...
    for (int iter = 0; iter < num_iters; iter ++) {
        if constexpr(sharded) {
            // do something...
            int i = 0;
            auto &shard = graph[pid];
            std::vector<MPI_Request> sendreqs;
            
            std::vector<int> sendbufs;
            sendbufs.resize(2 * shard.num_nodes());
            int n_updates = 0;
            int update_offset = 0;
            int prev = (pid - 1 + scheduler.num_colors) % scheduler.num_colors;
            int next = (pid + 1) % scheduler.num_colors;
            std::vector<int> recv_buf(2 * batch_size);
            std::vector<int> fw_updates(2 * batch_size);

            MPI_Request fw_req = MPI_REQUEST_NULL;
            MPI_Request send_req = MPI_REQUEST_NULL;
            MPI_Request recv_req = MPI_REQUEST_NULL;
            MPI_Status recv_status;
            int count;
            MPI_Irecv(recv_buf.data(), batch_size * 2, MPI_INT, prev, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_req);
            while (i < shard.num_nodes()) {
                int id = shard.local_to_global(i);
                int curr_color = scheduler.colors[id];
                if (curr_color != pid) {
                    i ++ ;
                    continue;
                }

                int best_color = curr_color;
                double best_span = *std::max_element(scheduler.runtime.begin(), scheduler.runtime.end());
                
                std::vector<double> cleared_runtime = scheduler.runtime;
                cleared_runtime[curr_color] -= shard.get_node_weight(id);
                std::vector<int> ins = shard.get_node_in_neighbors(id);
                std::vector<int> outs = shard.get_node_out_neighbors(id);
                for (const auto &in : ins) {
                    int in_color = scheduler.colors[in];
                    cleared_runtime[in_color] -= shard.get_edge_weight(in, id);
                }
                for (const auto &out: outs) {
                    int out_color = scheduler.colors[out];
                    cleared_runtime[curr_color] -= shard.get_edge_weight(id, out);
                }

                for (int new_color=0; new_color < scheduler.num_colors; new_color++) {
                    if (new_color == curr_color) continue;
                    
                    for (const auto &in : ins) {
                        int in_color = scheduler.colors[in];
                        cleared_runtime[in_color] += in_color == curr_color ? 0.0 : shard.get_edge_weight(in, id);
                    }

                    for (const auto &out : outs) {
                        int out_color = scheduler.colors[out];
                        cleared_runtime[curr_color] += out_color == curr_color ? 0.0 : shard.get_edge_weight(id, out);
                    }

                    double new_span = *std::max_element(cleared_runtime.begin(), cleared_runtime.end());

                    if (new_span < best_span) {
                        best_span = new_span;
                        best_color = new_color;
                    }
                }

                int update_id = id;
                i ++;

                if (best_color != curr_color) {
                    sendbufs.push_back(update_id);
                    sendbufs.push_back(best_color);
                    n_updates += 2;
                    
                }

                if (scheduler.num_colors > 1 && n_updates >= 2 * batch_size) {
                    MPI_Isend(sendbufs.data() + update_offset, 2 * batch_size, MPI_INT, next, pid, MPI_COMM_WORLD, &send_req);
                    n_updates -= 2 * batch_size;
                    update_offset += 2 * batch_size;
                    sendreqs.push_back(send_req);
                }

                int recv_flag = 0;
                
                if (recv_req != MPI_REQUEST_NULL) {
                    MPI_Test(&recv_req, &recv_flag, &recv_status);
                }

                if (recv_flag) {
                    MPI_Wait(&fw_req, MPI_STATUS_IGNORE);
                    std::swap(recv_buf, fw_updates);

                    MPI_Get_count(&recv_status, MPI_INT, &count);
                    
                    if (count > 0 && recv_status.MPI_TAG != pid) {
                        MPI_Isend(fw_updates.data(), count, MPI_INT, next, recv_status.MPI_TAG, MPI_COMM_WORLD, &fw_req); 
                    }

                    for (int j = 0; j < count; j += 2) {
                        int update_id = fw_updates[j];
                        int new_color = fw_updates[j + 1];
                        int old_color = scheduler.colors[update_id];
                        
                        std::vector<int> update_ins = shard.get_node_in_neighbors(update_id);
                        std::vector<int> update_outs = shard.get_node_out_neighbors(update_id);

                        
                        scheduler.colors[update_id] = new_color;

                        if (update_id >= shard.node_start && update_id < shard.node_end) {
                            scheduler.runtime[old_color] -= shard.get_node_weight(update_id);
                            scheduler.runtime[new_color] += shard.get_node_weight(update_id);
                        }
                        

                        for (const auto &update_in : update_ins) {
                            int in_color = scheduler.colors[update_in];
                            double edge_weight = shard.get_edge_weight(update_in, update_id);
                            scheduler.runtime[in_color] -= in_color == old_color ? 0 : edge_weight;
                            scheduler.runtime[in_color] += in_color == old_color ? 0 : edge_weight;
                        }
                        for (const auto &update_out : update_outs) {
                            int out_color = scheduler.colors[update_out];
                            double edge_weight = shard.get_edge_weight(update_id, update_out);
                            scheduler.runtime[old_color] -= out_color == old_color ? 0 : edge_weight;
                            scheduler.runtime[old_color] += out_color == old_color ? 0 : edge_weight;
                        }
                    }
                    if (count > 0 && i < shard.num_nodes()) {
                        MPI_Irecv(recv_buf.data(), batch_size * 2, MPI_INT, prev, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_req);
                    }
                }
            }
            MPI_Wait(&send_req, MPI_STATUS_IGNORE);
            MPI_Wait(&fw_req, MPI_STATUS_IGNORE);
            MPI_Isend(nullptr, 0, MPI_INT, next, pid, MPI_COMM_WORLD, &send_req);
            MPI_Wait(&send_req, MPI_STATUS_IGNORE);

            if (recv_req != MPI_REQUEST_NULL) {
                MPI_Wait(&recv_req, &recv_status);
                MPI_Get_count(&recv_status, MPI_INT, &count);
            } else {
                count = 0;
            }

            for (int j = 0; j < count; j += 2) {
                int update_id = recv_buf[j];
                int new_color = recv_buf[j + 1];
                int old_color = scheduler.colors[update_id];
                
                std::vector<int> update_ins = shard.get_node_in_neighbors(update_id);
                std::vector<int> update_outs = shard.get_node_out_neighbors(update_id);

                
                scheduler.colors[update_id] = new_color;
                if (update_id >= shard.node_start && update_id < shard.node_end) {
                    scheduler.runtime[old_color] -= shard.get_node_weight(update_id);
                    scheduler.runtime[new_color] += shard.get_node_weight(update_id);
                }
                

                for (const auto &update_in : update_ins) {
                    int in_color = scheduler.colors[update_in];
                    double edge_weight = shard.get_edge_weight(update_in, update_id);
                    scheduler.runtime[in_color] -= in_color == old_color ? 0 : edge_weight;
                    scheduler.runtime[in_color] += in_color == old_color ? 0 : edge_weight;
                }
                for (const auto &update_out : update_outs) {
                    int out_color = scheduler.colors[update_out];
                    double edge_weight = shard.get_edge_weight(update_id, update_out);
                    scheduler.runtime[old_color] -= out_color == old_color ? 0 : edge_weight;
                    scheduler.runtime[old_color] += out_color == old_color ? 0 : edge_weight;
                }
            }

        
        
        } else {
            // non-sharded case: single graph 
            int i = 0;
        
            std::vector<MPI_Request> sendreqs;
            
            std::vector<int> sendbufs;
            sendbufs.resize(2 * graph.num_nodes());
            int n_updates = 0;
            int update_offset = 0;
            int prev = (pid - 1 + scheduler.num_colors) % scheduler.num_colors;
            int next = (pid + 1) % scheduler.num_colors;
            std::vector<int> recv_buf(2 * batch_size);
            
            std::vector<int> fw_updates(2 * batch_size);

            MPI_Request fw_req = MPI_REQUEST_NULL;
            MPI_Request send_req = MPI_REQUEST_NULL;
            MPI_Request recv_req = MPI_REQUEST_NULL;
            MPI_Status recv_status;
            int count;
            MPI_Irecv(recv_buf.data(), batch_size * 2, MPI_INT, prev, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_req);
            while (i < graph.num_nodes()) {
                int curr_color = scheduler.colors[i];
                if (curr_color != pid) {
                    i ++ ;
                    continue;
                }

                int best_color = curr_color;
                double best_span = *std::max_element(scheduler.runtime.begin(), scheduler.runtime.end());
                
                std::vector<double> cleared_runtime = scheduler.runtime;
                cleared_runtime[curr_color] -= graph.get_node_weight(i);
                std::vector<int> ins = graph.get_node_in_neighbors(i);
                std::vector<int> outs = graph.get_node_out_neighbors(i);
                
                for (const auto &in : ins) {
                    int in_color = scheduler.colors[in];
                    cleared_runtime[in_color] -= graph.get_edge_weight(in, i);
                }
                for (const auto &out: outs) {
                    int out_color = scheduler.colors[out];
                    cleared_runtime[curr_color] -= graph.get_edge_weight(i, out);
                }

                for (int new_color=0; new_color < scheduler.num_colors; new_color++) {
                    if (new_color == curr_color) continue;
                    
                    for (const auto &in : ins) {
                        int in_color = scheduler.colors[in];
                        cleared_runtime[in_color] += in_color == curr_color ? 0.0 : graph.get_edge_weight(in, i);
                    }

                    for (const auto &out : outs) {
                        int out_color = scheduler.colors[out];
                        cleared_runtime[curr_color] += out_color == curr_color ? 0.0 : graph.get_edge_weight(i, out);
                    }

                    double new_span = *std::max_element(cleared_runtime.begin(), cleared_runtime.end());

                    if (new_span < best_span) {
                        best_span = new_span;
                        best_color = new_color;
                    }
                }

                int update_id = i;
                i ++;

                if (best_color != curr_color) {
                    sendbufs.push_back(update_id);
                    sendbufs.push_back(best_color);
                    n_updates += 2;
                    
                }

                if (scheduler.num_colors > 1 && n_updates >= 2 * batch_size) {
                    MPI_Isend(sendbufs.data() + update_offset, 2 * batch_size, MPI_INT, next, pid, MPI_COMM_WORLD, &send_req);
                    n_updates -= 2 * batch_size;
                    update_offset += 2 * batch_size;
                    sendreqs.push_back(send_req);
                }

                int recv_flag = 0;
                
                if (recv_req != MPI_REQUEST_NULL) {
                    MPI_Test(&recv_req, &recv_flag, &recv_status);
                }

                if (recv_flag) {
                    MPI_Wait(&fw_req, MPI_STATUS_IGNORE);
                    std::swap(recv_buf, fw_updates);

                    MPI_Get_count(&recv_status, MPI_INT, &count);

                    if (count > 0 && recv_status.MPI_TAG != pid) {
                        MPI_Isend(fw_updates.data(), count, MPI_INT, next, recv_status.MPI_TAG, MPI_COMM_WORLD, &fw_req); 
                    }

                    for (int j = 0; j < count; j += 2) {
                        int update_id = fw_updates[j];
                        int new_color = fw_updates[j + 1];
                        int old_color = scheduler.colors[update_id];
                        
                        std::vector<int> update_ins = graph.get_node_in_neighbors(update_id);
                        std::vector<int> update_outs = graph.get_node_out_neighbors(update_id);

                        
                        scheduler.colors[update_id] = new_color;
                        scheduler.runtime[old_color] -= graph.get_node_weight(update_id);
                        scheduler.runtime[new_color] += graph.get_node_weight(update_id);

                        for (const auto &update_in : update_ins) {
                            int in_color = scheduler.colors[update_in];
                            double edge_weight = graph.get_edge_weight(update_in, update_id);
                            scheduler.runtime[in_color] -= in_color == old_color ? 0 : edge_weight;
                            scheduler.runtime[in_color] += in_color == old_color ? 0 : edge_weight;
                        }
                        for (const auto &update_out : update_outs) {
                            int out_color = scheduler.colors[update_out];
                            double edge_weight = graph.get_edge_weight(update_id, update_out);
                            scheduler.runtime[old_color] -= out_color == old_color ? 0 : edge_weight;
                            scheduler.runtime[old_color] += out_color == old_color ? 0 : edge_weight;
                        }
                    }
                    if (count > 0 && i < graph.num_nodes()) {
                        MPI_Irecv(recv_buf.data(), batch_size * 2, MPI_INT, prev, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_req);
                    }
                }
            }
            MPI_Wait(&send_req, MPI_STATUS_IGNORE);
            MPI_Wait(&fw_req, MPI_STATUS_IGNORE);
            MPI_Isend(nullptr, 0, MPI_INT, next, pid, MPI_COMM_WORLD, &send_req);
            MPI_Wait(&send_req, MPI_STATUS_IGNORE);

            if (recv_req != MPI_REQUEST_NULL) {
                MPI_Wait(&recv_req, &recv_status);
                MPI_Get_count(&recv_status, MPI_INT, &count);
            } else {
                count = 0;
            }

            for (int j = 0; j < count; j += 2) {
                int update_id = recv_buf[j];
                int new_color = recv_buf[j + 1];
                int old_color = scheduler.colors[update_id];
                
                std::vector<int> update_ins = graph.get_node_in_neighbors(update_id);
                std::vector<int> update_outs = graph.get_node_out_neighbors(update_id);

                
                scheduler.colors[update_id] = new_color;
                scheduler.runtime[old_color] -= graph.get_node_weight(update_id);
                scheduler.runtime[new_color] += graph.get_node_weight(update_id);

                for (const auto &update_in : update_ins) {
                    int in_color = scheduler.colors[update_in];
                    double edge_weight = graph.get_edge_weight(update_in, update_id);
                    scheduler.runtime[in_color] -= in_color == old_color ? 0 : edge_weight;
                    scheduler.runtime[in_color] += in_color == old_color ? 0 : edge_weight;
                }
                for (const auto &update_out : update_outs) {
                    int out_color = scheduler.colors[update_out];
                    double edge_weight = graph.get_edge_weight(update_id, update_out);
                    scheduler.runtime[old_color] -= out_color == old_color ? 0 : edge_weight;
                    scheduler.runtime[old_color] += out_color == old_color ? 0 : edge_weight;
                }
            }

        }
    }
    update_scheduler_runtime<sharded>(graph, scheduler);
}
