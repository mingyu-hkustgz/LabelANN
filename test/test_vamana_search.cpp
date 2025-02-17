#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <boost/program_options.hpp>
#include "vamana/vamana.h"
#include "utils.h"

namespace po = boost::program_options;



int main(int argc, char** argv) {
    std::string data_type, dist_fn, base_bin_file, query_bin_file, base_label_file, query_label_file, gt_file, index_path_prefix;
    ANNS::IdxType K, Lsearch;
    uint32_t num_threads;

    try {
        po::options_description desc{"Arguments"};
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                           "data type <int8/uint8/float>");
        desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                           "distance function <L2/IP/cosine>");
        desc.add_options()("base_bin_file", po::value<std::string>(&base_bin_file)->required(),
                           "File containing the base vectors in binary format");
        desc.add_options()("query_bin_file", po::value<std::string>(&query_bin_file)->required(),
                           "File containing the query vectors in binary format");
        desc.add_options()("base_label_file", po::value<std::string>(&base_label_file)->default_value(""),
                           "Base label file in txt format");
        desc.add_options()("query_label_file", po::value<std::string>(&query_label_file)->default_value(""),
                           "Query label file in txt format");
        desc.add_options()("gt_file", po::value<std::string>(&gt_file)->required(),
                           "Filename for the writing ground truth in binary format");
        desc.add_options()("K", po::value<ANNS::IdxType>(&K)->required(),
                           "Number of ground truth nearest neighbors to compute");
        desc.add_options()("num_threads", po::value<uint32_t>(&num_threads)->default_value(ANNS::default_paras::NUM_THREADS),
                           "Number of threads to use");

        // vamana parameters
        desc.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                           "Prefix of the path to load the index");
        desc.add_options()("Lsearch", po::value<ANNS::IdxType>(&Lsearch)->default_value(ANNS::default_paras::L_SEARCH),
                           "Number of candidates to search in the graph");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        return -1;
    }

    // load base and query data
    std::shared_ptr<ANNS::IStorage> base_storage = ANNS::create_storage(data_type);
    std::shared_ptr<ANNS::IStorage> query_storage = ANNS::create_storage(data_type);
    base_storage->load_from_file(base_bin_file, base_label_file);
    query_storage->load_from_file(query_bin_file, query_label_file);

    // load index
    std::shared_ptr<ANNS::Graph> graph = std::make_shared<ANNS::Graph>(base_storage->get_num_points());
    ANNS::Vamana index;
    index.load(index_path_prefix, graph);

    // preparation
    auto num_queries = query_storage->get_num_points();
    std::shared_ptr<ANNS::DistanceHandler> distance_handler = ANNS::get_distance_handler(data_type, dist_fn);
    auto gt = new std::pair<ANNS::IdxType, float>[num_queries * K];
    load_gt_file(gt_file, gt, num_queries, K);
    auto results = new std::pair<ANNS::IdxType, float>[num_queries * K];
    std::vector<ANNS::IdxType> num_cmps(num_queries);


    std::vector efSearch{1, 2, 4, 8, 16, 32, 50, 64, 128, 150, 256, 300};
    std::ofstream fout("./results/sift/sift-vamana.log");
    for (auto ef : efSearch)
    {
        // search
        if(K > ef) ef = K;
        std::cout << "Start querying ..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        index.search(base_storage, query_storage, distance_handler, K, ef, num_threads, results, num_cmps);
        auto time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count();

        // statistics
        std::cout << "- Time cost: " << time_cost << "ms" << std::endl;
        std::cout << "- QPS: " << num_queries * 1000.0 / time_cost << std::endl;
        float total_cmps = std::accumulate(num_cmps.begin(), num_cmps.end(), 0);
        std::cout << "- Average number of comparisons: " << total_cmps / num_queries << std::endl;
        // calculate recall
        auto recall = calculate_recall(gt, results, num_queries, K);
        std::cout << "- Recall: " << recall << "%" << std::endl;
        fout<<recall<<" "<<num_queries * 1000.0 / time_cost<<std::endl;
    }

    return 0;
}