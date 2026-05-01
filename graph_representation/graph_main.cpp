#include "graph.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>

int main(int argc, char *argv[]) {
    std::filesystem::path raw_path = "../data_generation/raw_dataset";
    std::vector<std::filesystem::path> raw_files;
    for (const auto &f : std::filesystem::directory_iterator(raw_path)) { 
        raw_files.push_back(f.path());
    }

    int processed_graphs = 0;

    for (const auto &f : raw_files) {
        
        RawGraph raw_graph;
        load_raw(raw_graph, f);

        std::vector<ShardedGraph> all_sharded;
        raw_to_sharded(raw_graph, all_sharded); 

        const std::string s_sharded = "../dataset/sharded/" + f.stem().string() + "/";
        save_all_sharded(all_sharded, s_sharded);

        CompressedGraph compressed;
        const std::string s_compressed = "../dataset/compressed/" + f.stem().string() + "/";
        raw_to_compressed(raw_graph, compressed);
        save_compressed(compressed, s_compressed);
        
        std::vector<CompressedShardedGraph> all_compressed_sharded(all_sharded.size());
        for (int i = 0; i < all_sharded.size(); i++) {
            sharded_to_compressedsharded(all_sharded[i], all_compressed_sharded[i]);
        }

        const std::string s_compressed_sharded = "../dataset/compressed_sharded/" + f.stem().string() + "/";
        save_all_compressed_sharded(all_compressed_sharded, s_compressed_sharded);

        processed_graphs++;
        

    }

    std::cout << "FINISHED processing " << processed_graphs << " graphs" << std::endl;

    
    

    return 0;
}
