//
// Created by bld on 25-2-14.
//
#include <iostream>
#include <fstream>
#include <random>
#include <math.h>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <boost/program_options.hpp>
#include "config.h"
#include "trie.h"

namespace po = boost::program_options;

uint8_t max_tries = 100;
std::random_device rd;
std::mt19937 gen(rd());



// check if there are K vectors with super label sets
bool has_k_answers(const ANNS::TrieIndex& trie_index, const std::vector<ANNS::LabelType>& label_set,
                   ANNS::IdxType K, const std::string& scenario) {

    // if is equality scenario, check if the label set exists and the group size
    if (scenario == "equality") {
        auto node = trie_index.find_exact_match(label_set);
        return node && (node->group_size >= K);
    }

    // get the candidate super sets
    std::vector<std::shared_ptr<ANNS::TrieNode>> super_set_entrances;
    trie_index.get_super_set_entrances(label_set, super_set_entrances, false, scenario=="containment");

    // push the candidate super sets into the queue
    ANNS::IdxType cnt = 0;
    std::queue<std::shared_ptr<ANNS::TrieNode>> q;
    for (const auto& node : super_set_entrances) {
        q.push(node);
        cnt += node->group_size;
        if (cnt >= K)
            return true;
    }

    while (!q.empty()) {
        auto cur = q.front();
        q.pop();

        // push each child into the queue
        for (const auto& child : cur->children) {
            cnt += child.second->group_size;
            if (cnt >= K)
                return true;
            q.push(child.second);
        }
    }
    return false;
}



class ZipfDistribution {

public:
    ZipfDistribution(ANNS::IdxType num_points, ANNS::LabelType num_labels)
            : num_labels(num_labels), num_points(num_points),
              uniform_zero_to_one(std::uniform_real_distribution<>(0.0, 1.0)) {}


    // write the distribution to a file
    void write_distribution(const ANNS::TrieIndex& trie_index, ANNS::IdxType K, std::vector<std::vector<ANNS::LabelType>>& label_sets,
                            const std::string& scenario) {
        auto distribution_map = create_distribution_map();

        // assign label to each vector
        for (ANNS::IdxType i=0; i < num_points; i++) {
            std::vector<ANNS::LabelType> label_set;

            // try to find a valid label set
            for (auto j=0; j<max_tries; ++j) {

                // try each label
                for (ANNS::LabelType label=1; label<=num_labels; ++label) {
                    auto label_selection_probability = std::bernoulli_distribution(distribution_factor / (double)label);
                    if (label_selection_probability(rand_engine) && distribution_map[label] > 0)
                        label_set.emplace_back(label);
                }

                // check if the label set is valid
                if (has_k_answers(trie_index, label_set, K, scenario)) {
                    for (auto label : label_set)
                        distribution_map[label] -= 1;
                    break;
                }
                label_set.clear();
            }

            // when no valid labels exist, sample one from the cached label sets, preserve the distribution
            if (label_set.empty()) {
                if (label_sets.empty()) {
                    std::cerr << "Error: no valid label set found for " << scenario << " scenario with " << max_tries << " tries!" << std::endl;
                    exit(-1);
                }
                std::uniform_int_distribution<> dis(0, label_sets.size() - 1);
                label_set = label_sets[dis(gen)];
            }
            label_sets.emplace_back(label_set);
        }
    }

private:
    const ANNS::LabelType num_labels;
    const ANNS::IdxType num_points;
    const double distribution_factor = 0.7;
    std::knuth_b rand_engine;
    const std::uniform_real_distribution<double> uniform_zero_to_one;

    // compute the frequency of each label
    std::vector<ANNS::IdxType> create_distribution_map() {
        std::vector<ANNS::IdxType> distribution_map(num_labels + 1, 0);
        auto primary_label_freq = (ANNS::IdxType)ceil(num_points * distribution_factor);
        for (ANNS::LabelType i=1; i < num_labels + 1; i++)
            distribution_map[i] = (ANNS::IdxType)ceil(primary_label_freq / i);
        return distribution_map;
    }
};



int main(int argc, char **argv) {
    std::string input_file, output_file, distribution_type, scenario;
    ANNS::IdxType num_points, K, expected_num_label, max_num_label;
    try {
        po::options_description desc{"Arguments"};

        desc.add_options()("help", "Print information on arguments");
        desc.add_options()("input_file", po::value<std::string>(&input_file)->required(),
                           "Filename for the base label file");
        desc.add_options()("output_file", po::value<std::string>(&output_file)->required(),
                           "Filename for saving the label file");
        desc.add_options()("num_points", po::value<ANNS::IdxType>(&num_points)->required(), "Number of points in dataset");
        desc.add_options()("distribution_type", po::value<std::string>(&distribution_type)->default_value("random"),
                           "Distribution function for labels <random/zipf/uniform/poisson/one_per_point>, \
                           defaults to random");
        desc.add_options()("K", po::value<ANNS::IdxType>(&K)->default_value(10),
                           "At least K vectors with super label sets");
        desc.add_options()("scenario", po::value<std::string>(&scenario)->default_value("containment"),
                           "Type of filter scenario <containment/equality/overlap>");
        desc.add_options()("expected_num_label", po::value<ANNS::IdxType>(&expected_num_label)->default_value(3),
                           "Expected number of labels per point for multi-normial distribution");
        desc.add_options()("max_num_label", po::value<ANNS::IdxType>(&max_num_label)->default_value(12),
                           "Maximum number of labels per point for uniform distribution");

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

    // read base vector labels
    ANNS::TrieIndex trie_index;
    std::ifstream infile(input_file);
    if (!infile.is_open()) {
        std::cerr << "Error: could not open input file " << input_file << std::endl;
        return -1;
    }

    // build trie tree index
    ANNS::IdxType new_label_set_id = 1;
    std::string line, label;
    while (std::getline(infile, line)) {
        std::vector<ANNS::LabelType> label_set;
        std::stringstream ss(line);
        while (std::getline(ss, label, ','))
            label_set.emplace_back(std::stoi(label));
        trie_index.insert(label_set, new_label_set_id);
    }
    std::cout << "There are " << new_label_set_id - 1 << " unique base label sets" << std::endl;

    // prepare
    std::ofstream outfile(output_file);
    if (!outfile.is_open()) {
        std::cerr << "Error: could not open output file " << output_file << std::endl;
        return -1;
    }
    std::vector<std::vector<ANNS::LabelType>> label_sets;
    ANNS::LabelType num_labels = trie_index.get_max_label_id();
    std::cout << "Generating query label sets from " << num_labels << " possible labels for " << scenario << " scenario ..." << std::endl;

    // zipf distribution
    if (distribution_type == "zipf") {
        ZipfDistribution zipf(num_points, num_labels);
        zipf.write_distribution(trie_index, K, label_sets, scenario);

        // multi-normial distribution, expected_num_label / num_label chance to assign each label
    } else if (distribution_type == "multi_normial") {
        for (ANNS::IdxType i = 0; i < num_points; i++) {
            std::vector<ANNS::LabelType> label_set;

            // try to find a valid label set
            for (auto j=0; j<max_tries; ++j) {

                // try each label
                for (ANNS::LabelType label = 1; label <= num_labels; label++)
                    if (float(rand()) / RAND_MAX < float(expected_num_label) / num_labels)
                label_set.emplace_back(label);

                // check if the label set is valid
                if (has_k_answers(trie_index, label_set, K, scenario))
                    break;
                label_set.clear();
            }

            // when no valid labels exist, sample one from the cached label sets, preserve the distribution
            if (label_set.empty()) {
                if (label_sets.empty()) {
                    std::cerr << "Error: no valid label set found for " << scenario << " scenario with " << max_tries << " tries!" << std::endl;
                    exit(-1);
                }
                std::uniform_int_distribution<> dis(0, label_sets.size() - 1);
                label_set = label_sets[dis(gen)];
            }
            label_sets.emplace_back(label_set);
        }

        // uniform distribution
    } else if (distribution_type == "uniform") {
        std::uniform_int_distribution<> distr(1, num_labels); // define the range

        for (size_t i = 0; i < num_points; i++) {
            ANNS::IdxType num_labels_cur_point = rand() % (expected_num_label * 2) + 1;
            std::vector<ANNS::LabelType> label_set;

            // try to find a valid label set
            for (auto j=0; j<max_tries; ++j) {
                while (label_set.size() < num_labels_cur_point) {
                    ANNS::LabelType label = distr(gen);
                    if (std::find(label_set.begin(), label_set.end(), label) == label_set.end())
                        label_set.emplace_back(label);
                }

                // check if the label set is valid
                std::sort(label_set.begin(), label_set.end());
                if (has_k_answers(trie_index, label_set, K, scenario))
                    break;
                label_set.clear();
            }

            // when no valid labels exist, sample one from the cached label sets, preserve the distribution
            if (label_set.empty()) {
                if (label_sets.empty()) {
                    std::cerr << "Error: no valid label set found for " << scenario << " scenario with " << std::to_string(max_tries) << " tries!" << std::endl;
                    exit(-1);
                }
                std::uniform_int_distribution<> dis(0, label_sets.size() - 1);
                label_set = label_sets[dis(gen)];
            }
            label_sets.emplace_back(label_set);
        }

        // poisson distribution
    } else if (distribution_type == "poisson") {
        std::poisson_distribution<> distr(expected_num_label); // define the range

        for (size_t i = 0; i < num_points; i++) {
            ANNS::IdxType num_labels_cur_point = distr(gen) % num_labels + 1;
            std::vector<ANNS::LabelType> label_set;

            // try to find a valid label set
            for (auto j=0; j<max_tries; ++j) {
                while (label_set.size() < num_labels_cur_point) {
                    ANNS::LabelType label = distr(gen) % num_labels + 1;
                    if (std::find(label_set.begin(), label_set.end(), label) == label_set.end())
                        label_set.emplace_back(label);
                }

                // check if the label set is valid
                std::sort(label_set.begin(), label_set.end());
                if (has_k_answers(trie_index, label_set, K, scenario))
                    break;
                label_set.clear();
            }

            // when no valid labels exist, sample one from the cached label sets, preserve the distribution
            if (label_set.empty()) {
                if (label_sets.empty()) {
                    std::cerr << "Error: no valid label set found for " << scenario << " scenario with " << std::to_string(max_tries) << " tries!" << std::endl;
                    exit(-1);
                }
                std::uniform_int_distribution<> dis(0, label_sets.size() - 1);
                label_set = label_sets[dis(gen)];
            }
            label_sets.emplace_back(label_set);
        }

        // each point has only one label
    } else if (distribution_type == "one_per_point") {
        std::uniform_int_distribution<> distr(0, num_labels);
        for (size_t i = 0; i < num_points; i++) {
            std::vector<ANNS::LabelType> label_set;

            // try to find a valid label set
            for (auto j=0; j<max_tries; ++j) {
                label_set.emplace_back(distr(gen));

                // check if the label set is valid
                if (has_k_answers(trie_index, label_set, K, scenario))
                    break;
                label_set.clear();
            }

            // when no valid labels exist, sample one from the cached label sets, preserve the distribution
            if (label_set.empty()) {
                if (label_sets.empty()) {
                    std::cerr << "Error: no valid label set found for " << scenario << " scenario with " << max_tries << " tries!" << std::endl;
                    exit(-1);
                }
                std::uniform_int_distribution<> dis(0, label_sets.size() - 1);
                label_set = label_sets[dis(gen)];
            }
            label_sets.emplace_back(label_set);
        }
    }

    // write the labels to the file
    for (auto label_set : label_sets) {
        for (ANNS::LabelType i=0; i < label_set.size()-1; i++)
            outfile << label_set[i] << ',';
        outfile << label_set[label_set.size()-1] << std::endl;
    }
    outfile.close();
    std::cout << "Labels written to " << output_file << std::endl;
    return 0;
}