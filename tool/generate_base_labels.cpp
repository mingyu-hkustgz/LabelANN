#include <iostream>
#include <fstream>
#include <random>
#include <math.h>
#include <cmath>
#include <boost/program_options.hpp>
#include "config.h"

namespace po = boost::program_options;

std::random_device rd;
std::mt19937 gen(rd());


class ZipfDistribution {

public:
    ZipfDistribution(ANNS::IdxType num_points, ANNS::LabelType num_labels)
            : num_labels(num_labels), num_points(num_points),
              uniform_zero_to_one(std::uniform_real_distribution<>(0.0, 1.0)) {}


    // write the distribution to a file
    void write_distribution(std::ofstream& outfile) {
        auto distribution_map = create_distribution_map();
        std::vector<std::vector<ANNS::LabelType>> label_sets;

        // assign label to each vector
        for (ANNS::IdxType i=0; i < num_points; i++) {
            std::vector<ANNS::LabelType> label_set;

            // try each label
            for (ANNS::LabelType label=1; label<=num_labels; ++label) {
                auto label_selection_probability = std::bernoulli_distribution(distribution_factor / (double)label);
                if (label_selection_probability(rand_engine) && distribution_map[label] > 0) {
                    label_set.emplace_back(label);
                    distribution_map[label] -= 1;
                }
            }

            // when no valid labels exist, sample one from the cached label sets, preserve the distribution
            if (label_set.empty()) {
                std::uniform_int_distribution<> dis(0, label_sets.size() - 1);
                label_set = label_sets[dis(gen)];
            }
            label_sets.emplace_back(label_set);
        }

        // write the labels to the file
        for (auto label_set : label_sets) {
            for (ANNS::LabelType i=0; i < label_set.size()-1; i++)
                outfile << label_set[i] << ',';
            outfile << label_set[label_set.size()-1] << std::endl;
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
    std::string output_file, distribution_type(argv[2]);
    ANNS::LabelType num_labels;
    ANNS::IdxType num_points, expected_num_label, max_num_label;;
    try {
        po::options_description desc{"Arguments"};

        desc.add_options()("help", "Print information on arguments");
        desc.add_options()("output_file", po::value<std::string>(&output_file)->required(),
                           "Filename for saving the label file");
        desc.add_options()("num_points", po::value<ANNS::IdxType>(&num_points)->required(),
                           "Number of points in dataset");
        desc.add_options()("num_labels", po::value<ANNS::LabelType>(&num_labels)->required(),
                           "Number of unique labels");
        desc.add_options()("distribution_type", po::value<std::string>(&distribution_type)->default_value("zipf"),
                           "Distribution function for labels <multi_normial/zipf/uniform/poisson/one_per_point>, \
                           defaults to zipf");
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

    // open file
    std::ofstream outfile(output_file);
    if (!outfile.is_open()) {
        std::cerr << "Error: could not open output file " << output_file << std::endl;
        return -1;
    }

    // zipf distribution
    if (distribution_type == "zipf") {
        ZipfDistribution zipf(num_points, num_labels);
        zipf.write_distribution(outfile);

        // multi-normial distribution, expected_num_label / num_label chance to assign each label
    } else if (distribution_type == "multi_normial") {
        for (ANNS::IdxType i = 0; i < num_points; i++) {
            bool label_written = false;

            // try each label
            while (!label_written) {
                for (ANNS::LabelType label = 1; label <= num_labels; label++) {
                    if (float(rand()) / RAND_MAX < float(expected_num_label) / num_labels) {
                        if (label_written)
                            outfile << ',';
                        outfile << label;
                        label_written = true;
                    }
                }
            }
            outfile << std::endl;
        }

        // uniform distribution
    } else if (distribution_type == "uniform") {
        std::uniform_int_distribution<> distr(1, num_labels); // define the range

        for (size_t i = 0; i < num_points; i++) {
            ANNS::IdxType num_labels_cur_point = rand() % (expected_num_label * 2) + 1;
            std::vector<ANNS::LabelType> label_set;

            // assign labels
            while (label_set.size() < num_labels_cur_point) {
                ANNS::LabelType label = distr(gen);
                if (std::find(label_set.begin(), label_set.end(), label) == label_set.end())
                    label_set.emplace_back(label);
            }

            // write labels
            std::sort(label_set.begin(), label_set.end());
            for (size_t j = 0; j < num_labels_cur_point; j++) {
                if (j > 0)
                    outfile << ',';
                outfile << label_set[j];
            }
            outfile << std::endl;
        }

        // poisson distribution
    } else if (distribution_type == "poisson") {
        std::poisson_distribution<> distr(expected_num_label); // define the range

        for (size_t i = 0; i < num_points; i++) {
            ANNS::IdxType num_labels_cur_point = distr(gen) % num_labels + 1;
            std::vector<ANNS::LabelType> label_set;

            // assign labels
            while (label_set.size() < num_labels_cur_point) {
                ANNS::LabelType label = distr(gen) % num_labels + 1;
                if (std::find(label_set.begin(), label_set.end(), label) == label_set.end())
                    label_set.emplace_back(label);
            }

            // write labels
            std::sort(label_set.begin(), label_set.end());
            for (size_t j = 0; j < num_labels_cur_point; j++) {
                if (j > 0)
                    outfile << ',';
                outfile << label_set[j];
            }
            outfile << std::endl;
        }

        // each point has only one label
    } else if (distribution_type == "one_per_point") {
        std::uniform_int_distribution<> distr(0, num_labels); // define the range

        for (size_t i = 0; i < num_points; i++)
            outfile << distr(gen) << std::endl;
    }

    // close file
    outfile.close();
    std::cout << "Labels written to " << output_file << std::endl;
    return 0;
}