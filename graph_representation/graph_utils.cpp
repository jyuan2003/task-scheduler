#include "graph.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <filesystem>



void load_raw(RawGraph &raw, const std::string &fname) {
    int n;
    int m;  
    std::ifstream fin(fname);

    if (!fin) {
        std::cout << "FAILED to load raw graph " << fname << std::endl;
        return;
    }
    std::string _;
    
    fin >> _ >> n;

    raw.nodes.resize(n);

    for (int i = 0; i < n; i++) {
        int id;
        double weight;
        fin >> id >> weight;
        raw.nodes[id].id = id;
        raw.nodes[id].weight = weight;
    }

    fin >> _ >> m;
    raw.edges.resize(m);

    for (int i = 0; i < m; i++) {
        fin >> raw.edges[i].u >> raw.edges[i].v >> raw.edges[i].weight;
    }

    std::cout << "SUCCESSFULLY loaded raw graph" << " " << fname << std::endl;
       

}

void save_all_sharded(std::vector<ShardedGraph> &all_sharded, const std::string &prefix) {

    std::filesystem::create_directories(prefix);

    for (int i = 0; i < all_sharded.size(); i++) {
        const std::string fname = prefix + "sharded" + "_" + std::to_string(i) + ".txt";
        std::ofstream fout(fname);  

        if (!fout) {
            std::cout << "FAILED to write sharded graph to " << fname << std::endl;
            continue;
        }
        fout << std::setprecision(17);

        fout << all_sharded[i].node_start << " " << all_sharded[i].node_end << " "
             << all_sharded[i].node_weights.size() << " " << all_sharded[i].edges.size() << "\n";

        for (const auto &w : all_sharded[i].node_weights) {
            fout << w << " ";
        }
        fout << "\n";

        for (const auto &e : all_sharded[i].edges) {
            fout << e.u << " " << e.v << " " << e.weight << "\n";
        }

        fout.close();
    }

    
    std::cout << "SUCCESSFULLY saved all sharded graphs" << " " << prefix << std::endl;

}

void load_sharded(ShardedGraph &shard, const std::string &fname) {
    std::ifstream fin(fname);

    if (!fin) {
        std::cout << "FAILED to load sharded graph " << fname << std::endl;
        return;
    }
   
    int node_size;
    int edge_size;
    fin >> shard.node_start >> shard.node_end >> node_size >> edge_size;

    shard.node_weights.resize(node_size);
    for (int i = 0; i < node_size; i++) {
        fin >> shard.node_weights[i];
    }


    shard.edges.resize(edge_size);

    for (int j = 0; j < edge_size; j ++) {
        fin >> shard.edges[j].u >> shard.edges[j].v >> shard.edges[j].weight;
    }
    
    

    fin.close();

    std::cout << "SUCCESSFULLY loaded sharded graph" << " " << fname << std::endl;

}


void raw_to_sharded(RawGraph &raw, std::vector<ShardedGraph> &all_sharded, int num_shards) {

    int interval_true_size;
    // int num_shards = (raw.nodes.size() + ShardSize - 1) / ShardSize;
    int shard_size = (raw.nodes.size() + num_shards - 1) / num_shards;
    all_sharded.resize(num_shards);
    int interval_id = 0;
    for (int interval_start = 0; interval_start < raw.nodes.size(); interval_start += shard_size) {
        interval_true_size = std::min(shard_size, (int)raw.nodes.size() - interval_start);

        all_sharded[interval_id].node_start = interval_start;
        all_sharded[interval_id].node_end = interval_start + interval_true_size;
        all_sharded[interval_id].node_weights.resize(interval_true_size);
        for (int offset = 0; offset < interval_true_size; offset++) {
            all_sharded[interval_id].node_weights[offset] = raw.nodes[interval_start + offset].weight;
        }
        interval_id ++;
    }

    for (const auto &e : raw.edges) {
        int dest = e.v / shard_size;
        all_sharded[dest].edges.push_back(e);

    }

    for (int i = 0; i < interval_id; i++) {
        std::sort(
            all_sharded[i].edges.begin(),
            all_sharded[i].edges.end(),
            [](const RawEdge &e1, const RawEdge &e2) {
                return e1.u < e2.u;
            });
    }

}

int get_min_bytes(int x) {
    if (x >= -128 && x <= 127) return 1;
    if (x >= -32768 && x <= 32767) return 2;
    if (x >= -8388608 && x <= 8388607) return 3;
    return 4;
}


void raw_to_compressed(RawGraph &raw, CompressedGraph &compressed) {
    
    compressed.nodes.resize(raw.nodes.size());
    for (int i = 0; i < raw.nodes.size(); i++) {
        compressed.nodes[i].id = i;
        compressed.nodes[i].weight = raw.nodes[i].weight;
    }

    std::vector<std::vector<int>> raw_out_neighbors(raw.nodes.size());
    std::vector<std::vector<double>> raw_weights(raw.nodes.size());

    for (const auto &e : raw.edges) {
        raw_out_neighbors[e.u].push_back(e.v);
        raw_weights[e.u].push_back(e.weight);
    }

    
    for (int i = 0; i < raw.nodes.size(); i++) {
        std::vector<int> out_neighbors_temp;
        std::vector<double> weights_temp;

        std::vector<int> indices(raw_out_neighbors[i].size());
        for (int ii = 0; ii < raw_out_neighbors[i].size(); ii++) {
            indices[ii] = ii;
        }
        std::sort(
            indices.begin(),
            indices.end(),
            [&] (const auto &i1, const auto &i2) {
                return raw_out_neighbors[i][i1] < raw_out_neighbors[i][i2];
            }
        );
        
        for (const auto &ii : indices) {
            out_neighbors_temp.push_back(raw_out_neighbors[i][ii]);
            weights_temp.push_back(raw_weights[i][ii]);
        }

        std::swap(out_neighbors_temp, raw_out_neighbors[i]);
        std::swap(weights_temp, raw_weights[i]);

        
        for (int j = raw_out_neighbors[i].size() - 1; j > 0; j--) {
            raw_out_neighbors[i][j] -= raw_out_neighbors[i][j - 1];
            
        }
        if (raw_out_neighbors[i].size() > 0) {
            raw_out_neighbors[i][0] -= i; 

        }
        CompressedGroup group;
        
        for (int j = 0; j < raw_out_neighbors[i].size(); j++) {
            if (j == 0) {
                group.header.bytes = get_min_bytes(raw_out_neighbors[i][j]);
                group.out_neighbors.push_back(raw_out_neighbors[i][j]);
                group.weights.push_back(raw_weights[i][j]);
            }
            else {
                if (group.header.bytes == get_min_bytes(raw_out_neighbors[i][j])) {
                    group.out_neighbors.push_back(raw_out_neighbors[i][j]);
                    group.weights.push_back(raw_weights[i][j]);
                }
                else {
                    group.header.num_edges = group.out_neighbors.size();
                    compressed.nodes[i].groups.push_back(group);

                    group = CompressedGroup();

                    group.header.bytes = get_min_bytes(raw_out_neighbors[i][j]);
                    group.out_neighbors.push_back(raw_out_neighbors[i][j]);
                    group.weights.push_back(raw_weights[i][j]);
                }
            }

        }

        if (!group.out_neighbors.empty()) {
            group.header.num_edges = group.out_neighbors.size();
            compressed.nodes[i].groups.push_back(group);
        }
        
        
    }
}

void save_compressed(CompressedGraph &compressed, const std::string &prefix) {


    std::filesystem::create_directories(prefix);

    const std::string fname = prefix + "compressed" + ".txt";
    std::ofstream fout(fname);

    if (!fout) {
        std::cout << "FAILED to write compressed graph to " << fname << std::endl;
        return;
    }

    fout << compressed.nodes.size() << "\n";

    for (int i = 0; i < compressed.nodes.size(); i++) {
        
        fout << compressed.nodes[i].id << " " << compressed.nodes[i].weight << " "
             << compressed.nodes[i].groups.size() << "\n";
        for (const auto &group : compressed.nodes[i].groups) {
            fout << (int)group.header.bytes << " " << group.header.num_edges << "\n";
            for (const auto &v : group.out_neighbors) {
                fout << v << " ";
            }
            fout << "\n";

            for (const auto &w : group.weights) {
                fout << w << " ";
            }
            fout << "\n";
        }
    }

    fout.close();

    std::cout << "SUCCESSFULLY saved compressed graph to" << " " << fname << std::endl;
}


void load_compressed(CompressedGraph &compressed, const std::string &fname) {
    std::ifstream fin(fname);

    if (!fin) {
        std::cout << "FAILED to load compressed graph " << fname << std::endl;
        return;
    }

    int num_nodes;

    fin >> num_nodes;

    compressed.nodes.resize(num_nodes);

    for (int i = 0; i < num_nodes; i++) {
        int group_size;
        fin >> compressed.nodes[i].id >> compressed.nodes[i].weight >> group_size;
        compressed.nodes[i].groups.resize(group_size);

        for (int j = 0; j < group_size; j++) {
            int num_edges;
            int bytes;
            fin >> bytes >> num_edges;
            compressed.nodes[i].groups[j].header.bytes = static_cast<std::uint8_t>(bytes);
            compressed.nodes[i].groups[j].header.num_edges = num_edges;
            compressed.nodes[i].groups[j].out_neighbors.resize(num_edges);
            compressed.nodes[i].groups[j].weights.resize(num_edges);
            
            for (int jj = 0; jj < num_edges; jj++) {
                fin >> compressed.nodes[i].groups[j].out_neighbors[jj];
            }
            for (int jj = 0; jj < num_edges; jj++) {
                fin >> compressed.nodes[i].groups[j].weights[jj];
            }

        }
    }
    fin.close();

    std::cout << "SUCCESSFULLY loaded compressed graph " << " " << fname << std::endl;

        
}


void sharded_to_compressedsharded(ShardedGraph &sharded, CompressedShardedGraph &compressed_sharded) {
    int start = sharded.node_start;
    int end = sharded.node_end;

    compressed_sharded.node_start = start;
    compressed_sharded.node_end = end;

    compressed_sharded.nodes.resize(end - start);

    std::vector<std::vector<int>> in_neighbors(end - start);
    std::vector<std::vector<double>> in_weights(end - start);


    for (const auto &e : sharded.edges) {
        in_neighbors[e.v - start].push_back(e.u);
        in_weights[e.v - start].push_back(e.weight);

    }


    for (int i = start; i < end; i++) {
        compressed_sharded.nodes[i - start].id = i;
        compressed_sharded.nodes[i - start].weight = sharded.node_weights[i - start];


        std::vector<int> edges_temp;
        std::vector<double> weights_temp;

        std::vector<int> indices(in_neighbors[i - start].size());

        for (int ii = 0; ii < in_neighbors[i - start].size(); ii++) {
            indices[ii] = ii;
        }

        std::sort(
            indices.begin(),
            indices.end(),
            [&](const auto &i1, const auto &i2) {
                return in_neighbors[i - start][i1] < in_neighbors[i - start][i2];
            });
        
        for (int ii = 0; ii < in_neighbors[i - start].size(); ii++) {
            edges_temp.push_back(in_neighbors[i - start][indices[ii]]);
            weights_temp.push_back(in_weights[i - start][indices[ii]]);
        }

        for (int j = edges_temp.size() - 1; j > 0; j--) {
            edges_temp[j] -= edges_temp[j - 1];
        }

        if (!edges_temp.empty()) {
            edges_temp[0] -= i;
        }

        CompressedGroup group;
        
        for (int j = 0; j < in_neighbors[i - start].size(); j++) {
            if (j == 0) {
                group.header.bytes = get_min_bytes(edges_temp[j]);
                group.out_neighbors.push_back(edges_temp[j]);
                group.weights.push_back(weights_temp[j]);
            }
            else {
                if (group.header.bytes == get_min_bytes(edges_temp[j])) {
                    group.out_neighbors.push_back(edges_temp[j]);
                    group.weights.push_back(weights_temp[j]);
                }
                else {
                    group.header.num_edges = group.out_neighbors.size();
                    compressed_sharded.nodes[i - start].groups.push_back(group);

                    group = CompressedGroup();

                    group.header.bytes = get_min_bytes(edges_temp[j]);
                    group.out_neighbors.push_back(edges_temp[j]);
                    group.weights.push_back(weights_temp[j]);
                }
            }

        }

        if (!group.out_neighbors.empty()) {
            group.header.num_edges = group.out_neighbors.size();
            compressed_sharded.nodes[i-start].groups.push_back(group);
        }
        

    }


}

void save_all_compressed_sharded(std::vector<CompressedShardedGraph> &all_compressed_sharded, const std::string &prefix) {
    

    std::filesystem::create_directories(prefix);

    for (int i = 0; i < all_compressed_sharded.size(); i++) {
        const std::string fname = prefix + "compressed_sharded" + "_" + std::to_string(i) + ".txt";
        std::ofstream fout(fname);

        if (!fout) {
            std::cout << "FAILED to write compressed sharded graph to " << fname << std::endl;
            return;
        }

        auto &compressed_sharded = all_compressed_sharded[i];
        fout << compressed_sharded.node_start << " " << compressed_sharded.node_end << " " << compressed_sharded.nodes.size() << "\n";

        for (const auto &node : compressed_sharded.nodes) {
            fout << node.id << " " << node.weight << " " << node.groups.size() << "\n";
            for (const auto &group : node.groups) {
                fout << (int)group.header.bytes << " " << group.header.num_edges << "\n";
                
                for (const auto & u : group.out_neighbors) {
                    fout << u << " ";
                }

                fout << "\n";

                for (const auto & w : group.weights) {
                    fout << w << " ";
                }

                fout << "\n";
            }

        }
        fout.close();

        std::cout << "SUCCESSFULLY saved compressed sharded graph to" << " " << fname << std::endl;
    }


}


void load_compressed_sharded(CompressedShardedGraph &compressed_sharded, const std::string fname) {

    std::ifstream fin(fname);

    if (!fin) {
        std::cout << "FAILED to load compressed graph " << fname << std::endl;
        return;
    }

    int node_start, node_end, num_nodes;

    fin >> node_start >> node_end >> num_nodes; 

    compressed_sharded.node_start = node_start;
    compressed_sharded.node_end = node_end;
    compressed_sharded.nodes.resize(num_nodes);

    for (int i = 0; i < num_nodes; i++) {
        int group_size, id;
        double weight;
        fin >> id >> weight >> group_size;

        compressed_sharded.nodes[i].groups.resize(group_size);
        compressed_sharded.nodes[i].id = id;
        compressed_sharded.nodes[i].weight = weight;
        
        for (int j = 0; j < group_size; j++) {
            int num_edges;
            int bytes;
            fin >> bytes >> num_edges;
            compressed_sharded.nodes[i].groups[j].header.bytes = static_cast<std::uint8_t>(bytes);
            compressed_sharded.nodes[i].groups[j].header.num_edges = num_edges;
            compressed_sharded.nodes[i].groups[j].out_neighbors.resize(num_edges);
            compressed_sharded.nodes[i].groups[j].weights.resize(num_edges);
            
            for (int jj = 0; jj < num_edges; jj++) {
                fin >> compressed_sharded.nodes[i].groups[j].out_neighbors[jj];
            }
            for (int jj = 0; jj < num_edges; jj++) {
                fin >> compressed_sharded.nodes[i].groups[j].weights[jj];
            }
        }
    }

    fin.close();

    std::cout << "SUCCESSFULLY loaded compressed sharded graph " << " " << fname << std::endl;
}


int RawGraph::num_nodes() const {
    return nodes.size();
}

int RawGraph::num_edges() const {
    return edges.size();
}

int RawGraph::local_to_global(int i) const{
    return i;
}

double RawGraph::get_node_weight(int i) const {
    return nodes[i].weight;
}

double RawGraph::get_edge_weight(int u, int v) const {
    for (const auto &e : edges) {
        if (e.u == u && e.v == v) {
            return e.weight;
        }
    }
    return 0.0;
}


std::vector<int> RawGraph::get_node_in_neighbors(int u) const {
    std::vector<int> in_neighbors;
    for (const auto & e: edges) {
        if (e.v == u) {
            in_neighbors.push_back(e.u);
        }
    }
    return in_neighbors;
}


std::vector<int> RawGraph::get_node_out_neighbors(int u) const {
    std::vector<int> out_neighbors;
    for (const auto & e: edges) {
        if (e.u == u) {
            out_neighbors.push_back(e.v);
        }
    }
    return out_neighbors;
}

int ShardedGraph::num_nodes() const {
    return node_end - node_start;
}

int ShardedGraph::num_edges() const {
    return edges.size();
}

int ShardedGraph::local_to_global(int i) const {
    return i + node_start;
}

double ShardedGraph::get_node_weight(int i) const {
    return node_weights[i - node_start];
}

double ShardedGraph::get_edge_weight(int u, int v) const {
    for (const auto &e : edges) {
        if (e.u == u && e.v == v) {
            return e.weight;
        }
    }
    return 0.0;
}


std::vector<int> ShardedGraph::get_node_in_neighbors(int u) const {
    std::vector<int> in_neighbors;
    for (const auto & e: edges) {
        if (e.v == u) {
            in_neighbors.push_back(e.u);
        }
    }
    return in_neighbors;
}


std::vector<int> ShardedGraph::get_node_out_neighbors(int u) const {
    std::vector<int> out_neighbors;
    for (const auto & e: edges) {
        if (e.u == u) {
            out_neighbors.push_back(e.v);
        }
    }
    return out_neighbors;
}


int CompressedGraph::num_nodes() const {
    return nodes.size();
}

int CompressedGraph::num_edges() const {
    int num_edges = 0;
    for (const auto &node : nodes) {
        for (const auto &group : node.groups) {
            num_edges += group.header.num_edges;
        }
    }
    return num_edges;
}

int CompressedGraph::local_to_global(int i) const {
    return i;
}

double CompressedGraph::get_node_weight(int i) const {
    return nodes[i].weight;
}

double CompressedGraph::get_edge_weight(int u, int v) const {
    int out = u;
    for (int i = 0; i < nodes[u].groups.size(); i++) {
        for (int j = 0; j < nodes[u].groups[i].out_neighbors.size(); j++) {

            out += nodes[u].groups[i].out_neighbors[j];
            if (v == out) {
                return nodes[u].groups[i].weights[j];
            }
        }
    } 
    return 0.0;
}


std::vector<int> CompressedGraph::get_node_in_neighbors(int u) const {
    std::vector<int> in_neighbors;
    
    for (int v = 0; v < num_nodes(); v ++) { 
        int out = v;   
        for (int i = 0; i < nodes[v].groups.size(); i++) {
            for (int j = 0; j < nodes[v].groups[i].out_neighbors.size(); j++) {
                out += nodes[v].groups[i].out_neighbors[j];
                if (out == u) {
                    in_neighbors.push_back(v);
                }
            }
        }
    }
    
    
    return in_neighbors;
}



std::vector<int> CompressedGraph::get_node_out_neighbors(int u) const {
    std::vector<int> out_neighbors;
    int out = u;
    for (int i = 0; i < nodes[u].groups.size(); i++) {
        for (int j = 0; j < nodes[u].groups[i].out_neighbors.size(); j++) {
            out += nodes[u].groups[i].out_neighbors[j];
            out_neighbors.push_back(out);
        }
    }
    
    return out_neighbors;
}




int CompressedShardedGraph::num_nodes() const {
    return node_end - node_start;
}

int CompressedShardedGraph::num_edges() const {
    int num_edges = 0;
    for (const auto &node : nodes) {
        for (const auto &group : node.groups) {
            num_edges += group.header.num_edges;
        }
    }
    return num_edges;
}

int CompressedShardedGraph::local_to_global(int i) const {
    return i + node_start;
}

double CompressedShardedGraph::get_node_weight(int i) const {
    return nodes[i - node_start].weight;
}

double CompressedShardedGraph::get_edge_weight(int u, int v) const {
    
    int out = u;
    for (int i = 0; i < nodes[u - node_start].groups.size(); i++) {
        for (int j = 0; j < nodes[u - node_start].groups[i].out_neighbors.size(); j++) {

            out += nodes[u - node_start].groups[i].out_neighbors[j];
            if (v == out) {
                return nodes[u - node_start].groups[i].weights[j];
            }
        }
    } 
    return 0.0;
}

std::vector<int> CompressedShardedGraph::get_node_in_neighbors(int u) const {
    std::vector<int> in_neighbors;
    for (int v = node_start; v < node_end; v++) {
        int out = v;
        for (int i = 0; i < nodes[v - node_start].groups.size(); i++) {
            for (int j = 0; j < nodes[v - node_start].groups[i].out_neighbors.size(); j++) {
                out += nodes[v - node_start].groups[i].out_neighbors[j];
                if (out == u) {
                    in_neighbors.push_back(v);
                }
                
            }
        }
    
    }
    
    return in_neighbors;
}



std::vector<int> CompressedShardedGraph::get_node_out_neighbors(int u) const {
    std::vector<int> out_neighbors;
    int out = u;
    for (int i = 0; i < nodes[u - node_start].groups.size(); i++) {
        for (int j = 0; j < nodes[u - node_start].groups[i].out_neighbors.size(); j++) {
            out += nodes[u - node_start].groups[i].out_neighbors[j];
            out_neighbors.push_back(out);
        }
    }
    
    return out_neighbors;
}


