//
// Created by Mingyu YANG on 25-2-15.
//
#include <iostream>
#include <fstream>
#include <cstdio>
#include <getopt.h>
#include "IndexLabelOpt.h"

using namespace std;


int main(int argc, char *argv[]) {
    const struct option longopts[] = {
            // General Parameter
            {"help",    no_argument,       0, 'h'},

            // Indexing Path
            {"dataset", required_argument, 0, 'd'},
            {"source",  required_argument, 0, 's'},
    };

    int ind;
    int iarg = 0, K = 10, num_thread = 1;
    opterr = 1; //getopt error message (off: 0)

    char dataset[256] = "";
    char source[256] = "";
    char label[256] = "";
    char data_path[256] = "";
    char index_path[256] = "";
    char query_path[256] = "";
    char ground_path[256] = "";
    char base_label_path[256] = "";
    char query_label_path[256] = "";
    char result_path[256] = "";
    char file_type[256] = "fvecs";
    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:s:t:k:l:e:f:", longopts, &ind);
        switch (iarg) {
            case 'd':
                if (optarg) {
                    strcpy(dataset, optarg);
                }
                break;
            case 's':
                if (optarg) {
                    strcpy(source, optarg);
                }
                break;
            case 't':
                if (optarg) num_thread = atoi(optarg);
                break;
            case 'k':
                if (optarg) K = atoi(optarg);
                break;
            case 'l':
                if (optarg) {
                    strcpy(label, optarg);
                }
                break;
            case 'f':
                if (optarg) {
                    strcpy(file_type, optarg);
                }
                break;
        }
    }
    sprintf(query_path, "%s%s_query.%s", source, dataset, file_type);
    sprintf(data_path, "%s%s_base.%s", source, dataset, file_type);
    sprintf(index_path, "%s%s_opt.hnsw", source, dataset);
    sprintf(ground_path, "%s%s_gt_%s_containment.bin", source, dataset, label);
    sprintf(base_label_path, "%s%s_base_%s.txt", source, dataset, label);
    sprintf(query_label_path, "%s%s_query_%s_containment.txt", source, dataset, label);
    Matrix<float> X(data_path);
    Matrix<float> Q(query_path);
    auto gt = new std::pair<ANNS::IdxType, float>[Q.n * K];
    load_gt_file(ground_path, gt, Q.n, K);
    hnswlib::HierarchicalNSWStatic<float>::static_base_data_ = (char *) X.data;
    IndexLabelOpt hnsw_opt(X.n, X.d, X.data);
    hnsw_opt.load_base_label_bitmap(base_label_path);
    hnsw_opt.load_query_label_bitmap(query_label_path, Q.n);
    hnsw_opt.load_optimal_index(index_path);

    std::vector efSearch{1, 2, 4, 8, 16, 32, 50, 64, 128, 150, 256, 300};
    sprintf(result_path, "./results@%d/%s/%s-hnsw-%s-opt.log", K, dataset, dataset, label);
    std::ofstream fout(result_path);
    for (auto ef: efSearch) {
        // search
        if (K > ef) ef = K;
        std::cout << "Start querying ..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        auto results = hnsw_opt.contain_parallel_search(Q.data, K, ef, 1);
        auto time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count();

        // statistics
        std::cout << "- Time cost: " << time_cost << "ms" << std::endl;
        std::cout << "- QPS: " << Q.n * 1000.0 / time_cost << std::endl;
        // calculate recall
        auto recall = calculate_recall(gt, results, Q.n, K);
        std::cout << "- Recall: " << recall << "%" << std::endl;
        fout << recall << " " << Q.n * 1000.0 / time_cost << std::endl;
    }

    return 0;
}
