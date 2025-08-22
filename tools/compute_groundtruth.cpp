#include <chrono>
#include <fstream>
#include <iostream>
#include <boost/program_options.hpp>
#include "vamana/filtered_scan.h"
#include "utils.h"

namespace po = boost::program_options;



int main(int argc, char** argv) {
    std::string data_type, dist_fn, base_bin_file, query_bin_file, base_label_file, query_label_file, gt_file, scenario;
    ANNS::IdxType K;
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
        desc.add_options()("scenario", po::value<std::string>(&scenario)->default_value("containment"),
                           "Type of filter scenario <containment/equality/overlap/nofilter>");
        desc.add_options()("K", po::value<ANNS::IdxType>(&K)->required(),
                           "Number of ground truth nearest neighbors to compute");
        desc.add_options()("num_threads", po::value<uint32_t>(&num_threads)->default_value(ANNS::default_paras::NUM_THREADS),
                           "Number of threads to use");

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

    // preparation
    std::shared_ptr<ANNS::DistanceHandler> distance_handler = ANNS::get_distance_handler(data_type, dist_fn);
    auto groundtruth = new std::pair<ANNS::IdxType, float>[query_storage->get_num_points() * K];
    auto queries_selectivity = std::vector<double>();
    queries_selectivity.resize(query_storage->get_num_points());
    std::cout << "Computing ground truth using filter then bruteforce search ..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // run
    ANNS::FilteredScan algo;
    algo.run(base_storage, query_storage, distance_handler, scenario, num_threads, K, groundtruth, queries_selectivity.data());
    auto time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Time cost: " << time_cost << "ms" << std::endl;

    // write ground truth to file
    write_gt_file(gt_file, groundtruth, query_storage->get_num_points(), K);
    std::string queries_selectivity_file = gt_file + "_selectivity.txt";
    write_query_selectivity_file(queries_selectivity_file, queries_selectivity);

    return 0;
}