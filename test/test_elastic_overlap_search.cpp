//
// Created by Mingyu YANG on 25-2-24.
//
#include <iostream>
#include <fstream>
#include <cstdio>
#include <getopt.h>
#include "IndexLabelElastic.h"

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
    float elastic_factor = 0.5;
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
            case 'e':
                if (optarg) elastic_factor = atof(optarg);
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
#ifndef ID_COMPACT
    sprintf(index_path, "%s%s_elastic_%.2f.hnsw", source, dataset, elastic_factor);
#else
    sprintf(index_path, "%s%s_elastic_%.2f_compact.hnsw", source, dataset, elastic_factor);
#endif

    sprintf(ground_path, "%s%s_gt_%s_overlap.bin", source, dataset, label);
    sprintf(base_label_path, "%s%s_base_%s.txt", source, dataset, label);
    sprintf(query_label_path, "%s%s_query_%s_overlap.txt", source, dataset, label);
    Matrix<float> X(data_path);
    Matrix<float> Q(query_path);
    auto gt = new std::pair<ANNS::IdxType, float>[Q.n * K];
    load_gt_file(ground_path, gt, Q.n, K);
    hnswlib::HierarchicalNSWStatic<float>::static_base_data_ = (char *) X.data;
    IndexLabelElastic hnsw_elastic(X.n, X.d, X.data);
    hnsw_elastic.set_elastic_factor(elastic_factor);
    hnsw_elastic.load_base_label_bitmap(base_label_path);
    hnsw_elastic.load_query_label_bitmap(query_label_path, Q.n);
    hnsw_elastic.load_elastic_index(index_path);

    std::vector efSearch{10, 16, 32, 64, 128, 256, 275, 512, 1024};
#ifndef ID_COMPACT
    sprintf(result_path, "./results@%d/%s/%s-hnsw-%s-overlap-%.2f.log", K, dataset, dataset, label, elastic_factor);
    if(num_thread!=1)
        sprintf(result_path, "./results@%d/%s/%s-hnsw-%s-overlap-%.2f-%d.log", K, dataset, dataset, label, elastic_factor,num_thread);
    std::ofstream fout(result_path);
#else
    sprintf(result_path, "./results@%d/%s/%s-hnsw-%s-overlap-%.2f-compact.log", K, dataset, dataset, label,
            elastic_factor);
    if(num_thread!=1)
        sprintf(result_path, "./results@%d/%s/%s-hnsw-%s-overlap-%.2f-compact-%d.log", K, dataset, dataset, label, elastic_factor,num_thread);
    std::ofstream fout(result_path);
#endif
    for (auto ef: efSearch) {
        // search
        if (K > ef) ef = K;
        std::cout << "Start querying ..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        auto results = hnsw_elastic.overlap_naive_search(Q.data, K, ef);
        auto time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count();

        // statistics
        std::cout << "- Time cost: " << time_cost << "ms" << std::endl;
        std::cout << "- QPS: " << Q.n * 1000.0 / time_cost << std::endl;
        // calculate recall
        auto recall = calculate_recall(gt, results, Q.n, K);
        auto ave_ratio = getRatio(gt, results, Q.n, K);
        std::cout << "- Recall: " << recall << "%" << std::endl;
        std::cout << "- Ratio: " << ave_ratio <<std::endl;
        fout << recall << " " << Q.n * 1000.0 / time_cost<<" "<<ave_ratio<< std::endl;
        if(recall > 99.9) break;
    }

    return 0;
}
