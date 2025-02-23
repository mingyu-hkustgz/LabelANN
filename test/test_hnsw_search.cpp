//
// Created by mingyu on 25-2-15.
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
    char data_path[256] = "";
    char index_path[256] = "";
    char query_path[256] = "";
    char ground_path[256] = "";
    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:s:t:k:", longopts, &ind);
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
        }
    }
    sprintf(query_path, "%s%s_query.bin", source, dataset);
    sprintf(data_path, "%s%s_base.bin", source, dataset);
    sprintf(index_path, "%s%s_opt.hnsw", source, dataset);
    sprintf(ground_path, "%s%s_gt_12_labels_zipf_containment.bin", source, dataset);
    Matrix<float> X(data_path);
    Matrix<float> Q(query_path);
    auto gt = new std::pair<ANNS::IdxType, float>[Q.n * K];
    load_gt_file(ground_path, gt, Q.n, K);
    hnswlib::HierarchicalNSWStatic<float>::static_base_data_ = (char *) X.data;
    IndexLabelOpt hnsw_opt(X.n, X.d);
    hnsw_opt.load_base_label_bitmap("./DATA/sift/sift_base_12_labels_zipf.txt");
    hnsw_opt.load_query_label_bitmap("./DATA/sift/sift_query_12_labels_zipf_containment.txt", Q.n);
    hnsw_opt.load_optimal_index(index_path);
    hnsw_opt.data_ = X.data;

    std::vector efSearch{1, 2, 4, 8, 16, 32, 50, 64, 128, 150, 256, 300};
    std::ofstream fout("./results/sift/sift-hnsw-opt.log");
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
//        std::cerr<<gt[0].first<<" "<<gt[0].second<<" "<<results[0].first<<" "<<results[0].second<<" "<<sqr_dist(X.data + results[0].first * Q.d,Q.data, Q.d)<<std::endl;
//        std::cerr<<gt[1].first<<" "<<gt[1].second<<" "<<results[1].first<<" "<<results[1].second<<std::endl;
//        std::cerr<<gt[2].first<<" "<<gt[2].second<<" "<<results[2].first<<" "<<results[2].second<<std::endl;
//        std::cerr<<gt[3].first<<" "<<gt[3].second<<" "<<results[3].first<<" "<<results[3].second<<std::endl;
//        std::cerr<<gt[4].first<<" "<<gt[4].second<<" "<<results[4].first<<" "<<results[4].second<<std::endl;
//        std::cerr<<gt[5].first<<" "<<gt[5].second<<" "<<results[5].first<<" "<<results[5].second<<std::endl;
//        std::cerr<<gt[6].first<<" "<<gt[6].second<<" "<<results[6].first<<" "<<results[6].second<<std::endl;
//        std::cerr<<gt[7].first<<" "<<gt[7].second<<" "<<results[7].first<<" "<<results[7].second<<std::endl;
//        std::cerr<<gt[8].first<<" "<<gt[8].second<<" "<<results[8].first<<" "<<results[8].second<<std::endl;
//        std::cerr<<gt[9].first<<" "<<gt[9].second<<" "<<results[9].first<<" "<<results[9].second<<std::endl;
//        std::cerr<<gt[10].first<<" "<<gt[10].second<<" "<<results[10].first<<" "<<results[10].second<<" "<<sqr_dist(X.data + results[10].first * Q.d,Q.data +Q.d, Q.d)<<std::endl;
//        std::cerr<<gt[11].first<<" "<<gt[11].second<<" "<<results[11].first<<" "<<results[11].second<<std::endl;
//        std::cerr<<gt[12].first<<" "<<gt[12].second<<" "<<results[12].first<<" "<<results[12].second<<std::endl;
//        std::cerr<<gt[13].first<<" "<<gt[13].second<<" "<<results[13].first<<" "<<results[13].second<<std::endl;
//        std::cerr<<gt[14].first<<" "<<gt[14].second<<" "<<results[14].first<<" "<<results[14].second<<std::endl;
//        std::cerr<<gt[15].first<<" "<<gt[15].second<<" "<<results[15].first<<" "<<results[15].second<<std::endl;
//        std::cerr<<gt[16].first<<" "<<gt[16].second<<" "<<results[16].first<<" "<<results[16].second<<std::endl;
//        std::cerr<<gt[17].first<<" "<<gt[17].second<<" "<<results[17].first<<" "<<results[17].second<<std::endl;
//        std::cerr<<gt[18].first<<" "<<gt[18].second<<" "<<results[18].first<<" "<<results[18].second<<std::endl;
//        std::cerr<<gt[19].first<<" "<<gt[19].second<<" "<<results[19].first<<" "<<results[19].second<<std::endl;
        std::cout << "- Recall: " << recall << "%" << std::endl;
        fout << recall << " " << Q.n * 1000.0 / time_cost << std::endl;
    }

    return 0;
}
