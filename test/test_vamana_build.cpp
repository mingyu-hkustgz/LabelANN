#include <chrono>
#include <fstream>
#include <iostream>
#include <boost/program_options.hpp>
#include "vamana/vamana.h"

namespace po = boost::program_options;



int main(int argc, char** argv) {

    // common auguments
    std::string data_type, dist_fn, base_bin_file, base_label_file, index_path_prefix;
    uint32_t num_threads;

    // parameters for Vamana
    ANNS::IdxType max_degree, Lbuild;
    float alpha;

    try {
        po::options_description desc{"Arguments"};

        // common arguments
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                           "data type <int8/uint8/float>");
        desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                           "distance function <L2/IP/cosine>");
        desc.add_options()("base_bin_file", po::value<std::string>(&base_bin_file)->required(),
                           "File containing the base vectors in binary format");
        desc.add_options()("base_label_file", po::value<std::string>(&base_label_file)->default_value(""),
                           "Base label file in txt format");
        desc.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                           "Path prefix for saving the index");
        desc.add_options()("num_threads", po::value<uint32_t>(&num_threads)->default_value(ANNS::default_paras::NUM_THREADS),
                           "Number of threads to use");

        // parameters for graph indices
        desc.add_options()("max_degree", po::value<ANNS::IdxType>(&max_degree)->default_value(ANNS::default_paras::MAX_DEGREE),
                           "Max degree for building Vamana");
        desc.add_options()("Lbuild", po::value<uint32_t>(&Lbuild)->default_value(ANNS::default_paras::L_BUILD),
                           "Size of candidate set for building Vamana");
        desc.add_options()("alpha", po::value<float>(&alpha)->default_value(ANNS::default_paras::ALPHA),
                           "Alpha for building Vamana");

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
    base_storage->load_from_file(base_bin_file, base_label_file);
    std::shared_ptr<ANNS::DistanceHandler> distance_handler = ANNS::get_distance_handler(data_type, dist_fn);

    // build vamana index
    auto start_time = std::chrono::high_resolution_clock::now();
    std::shared_ptr<ANNS::Graph> graph = std::make_shared<ANNS::Graph>(base_storage->get_num_points());
    ANNS::Vamana index;
    index.build(base_storage, distance_handler, graph, max_degree, Lbuild, alpha, num_threads);
    std::cout << "Index time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

    // statistics and save index
    index.statistics();
    index.save(index_path_prefix);
    return 0;
}