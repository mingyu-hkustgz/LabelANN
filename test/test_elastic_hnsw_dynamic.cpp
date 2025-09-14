//
// Created by Mingyu YANG on 25-2-24.
//
#include <iostream>
#include <fstream>
#include <cstdio>
#include <getopt.h>
#include "IndexLabelDynamic.h"

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
    int iarg = 0, K = 10, base_num, update_num;
    float elastic_factor = 0.2;
    opterr = 1; //getopt error message (off: 0)

    char dataset[256] = "";
    char source[256] = "";
    char label[256] = "";
    char data_path[256] = "";
    char base_label_path[256] = "";
    char query_path[256] = "";
    char ground_path[256] = "";
    char query_label_path[256] = "";
    char file_type[256] = "fvecs";
    char result_path[256] = "";
    char logger_path[256] = "";
    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:s:e:l:b:", longopts, &ind);
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
            case 'l':
                if (optarg) {
                    strcpy(label, optarg);
                }
                break;
            case 'e':
                if (optarg) elastic_factor = atof(optarg);
                break;
            case 'b':
                if (optarg) base_num = atoi(optarg);
                break;
        }
    }
    sprintf(query_path, "%s%s_query.%s", source, dataset, file_type);
    sprintf(data_path, "%s%s_base.%s", source, dataset, file_type);
    sprintf(base_label_path, "%s%s_base_%s.txt", source, dataset, label);
    sprintf(logger_path,"./results/%s_dynamic_0-%d.log", dataset, base_num);
    sprintf(ground_path, "%s%s_gt_%s_containment.bin", source, dataset, label);
    sprintf(query_label_path, "%s%s_query_%s_containment.txt", source, dataset, label);
    Matrix<float> X(data_path);
    Matrix<float> Q(query_path);
    auto gt = new std::pair<ANNS::IdxType, float>[Q.n * K];

    load_gt_file(ground_path, gt, Q.n, K);
    hnswlib::HierarchicalNSWStatic<float>::static_base_data_ = (char *) X.data;
    IndexLabelDynamic hnsw_elastic(base_num, X.d, X.data);
    hnsw_elastic.load_base_label_bitmap(base_label_path);
    hnsw_elastic.load_query_label_bitmap(query_label_path, Q.n);
    hnsw_elastic.set_elastic_factor(elastic_factor);


    auto start_time = std::chrono::high_resolution_clock::now();
    hnsw_elastic.build_elastic_index(X);
    hnsw_elastic.save_log(logger_path);
    int update_points = (int) X.n - base_num;
    hnsw_elastic.update_index(update_points, base_label_path);
    cout<<"HNSW Based Finished"<<std::endl;
    auto time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time).count();
    // statistics
    std::cout << "\r-Base Time cost: " << time_cost << "ms" << std::endl;
    std::cout << "\r Update Points: "<< update_points <<std::endl;
    sprintf(logger_path,"./results/%s_dynamic_%d-%d.log", dataset, base_num, update_points + base_num);


    start_time = std::chrono::high_resolution_clock::now();
    hnsw_elastic.incremental_elastic_index();
    time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time).count();


    // statistics
    std::cout << "\r-Incremental Time cost: " << time_cost << "ms" << std::endl;
    hnsw_elastic.save_log(logger_path);


    std::vector efSearch{10, 16, 32, 64, 128, 256, 275, 512, 800, 900, 1024};
    sprintf(result_path, "./results@%d/%s/%s-hnsw-%s-dynamic-%.2f.log", K, dataset, dataset, label, elastic_factor);
    std::ofstream fout(result_path);

    for (auto ef: efSearch) {
        // search
        if (K > ef) ef = K;
        std::cout << "Start querying ..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        auto results = hnsw_elastic.contain_search(Q.data, K, ef);
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
