
#include <vector>
#include <cstdint>
#include <string>
struct RawNode {
    int id;
    double weight;
};

struct RawEdge {
    // e = u->v
    int u;
    int v;
    double weight;
};

struct RawGraph {
    std::vector<RawNode> nodes;
    std::vector<RawEdge> edges;
    
    int num_nodes() const;
    int num_edges() const;

    double get_node_weight(int id) const;
    double get_edge_weight(int u, int v) const;

    std::vector<int> get_node_out_neighbors(int id) const;

    int local_to_global(int id) const;
};



struct ShardedGraph {
    int node_start;
    int node_end;
    std::vector<double> node_weights;
    std::vector<RawEdge> edges;

    int num_nodes() const;
    int num_edges() const;

    double get_node_weight(int id) const;
    double get_edge_weight(int u, int v) const;

    std::vector<int> get_node_out_neighbors(int id) const;

    int local_to_global(int id) const;

};


struct CompressedGroupHeader {
    std::uint8_t bytes;
    int num_edges;
};

struct CompressedGroup {
    CompressedGroupHeader header;
    std::vector<int> out_neighbors; 
    std::vector<double> weights;
};

struct CompressedGroups {
    int id;
    double weight;
    std::vector<CompressedGroup> groups;
};


struct CompressedGraph {
    std::vector<CompressedGroups> nodes;

    int num_nodes() const;
    int num_edges() const;

    double get_node_weight(int id) const;
    double get_edge_weight(int u, int v) const;

    std::vector<int> get_node_out_neighbors(int id) const;

    int local_to_global(int id) const;

};

struct CompressedShardedGraph {
    int node_start;
    int node_end;
    std::vector<CompressedGroups> nodes;

    int num_nodes() const;
    int num_edges() const;

    double get_node_weight(int id) const;
    double get_edge_weight(int u, int v) const;

    std::vector<int> get_node_out_neighbors(int id) const;

    int local_to_global(int) const;


};

void load_raw(RawGraph &raw, const std::string &fname);


void load_sharded(ShardedGraph &sharded, const std::string &fname);
void save_all_sharded(std::vector<ShardedGraph> &all_sharded, const std::string &prefix);
void raw_to_sharded(RawGraph &raw, std::vector<ShardedGraph> &all_sharded); 


void save_compressed(CompressedGraph &compressed, const std::string &fname);
void load_compressed(CompressedGraph &compressed, const std::string &fname);
void raw_to_compressed(RawGraph &raw, CompressedGraph &compressed);


void sharded_to_compressedsharded(ShardedGraph &sharded, CompressedShardedGraph &compressed_sharded);
void save_all_compressed_sharded(std::vector<CompressedShardedGraph> &all_compressed_sharded, const std::string &prefix);
void load_compressed_sharded(CompressedShardedGraph &compressed_sharded, const std::string fname);
