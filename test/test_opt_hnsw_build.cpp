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
    int iarg = 0;
    opterr = 1; //getopt error message (off: 0)

    char dataset[256] = "";
    char source[256] = "";
    char label[256] = "";
    char data_path[256] = "";
    char index_path[256] = "";
    char label_path[256] = "";
    char file_type[256] = "fvecs";
    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:s:l:f:", longopts, &ind);
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
            case 'f':
                if (optarg) {
                    strcpy(file_type, optarg);
                }
                break;
        }
    }
    sprintf(data_path, "%s%s_base.%s", source, dataset, file_type);
//    sprintf(index_path, "%s%s_opt.hnsw", source, dataset);
//    sprintf(label_path, "%s%s_base_%s.txt", source, dataset, label);
    Matrix<float> X(data_path);
    hnswlib::HierarchicalNSWStatic<float>::static_base_data_ = (char *) X.data;
//    auto l2space = new hnswlib::L2Space(X.d);
//    auto appr_alg = new hnswlib::HierarchicalNSWStatic<float>(l2space, X.n/2, 16, 200);
//    auto start_time = std::chrono::high_resolution_clock::now();
//#pragma omp parallel for schedule(dynamic, 144)
//    for (int i = 0; i < 500000; i++) {
//        appr_alg->addPoint(X.data + i * X.d, i);
//    }
//    cout<<"finished half"<<std::endl;
//    auto time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(
//            std::chrono::high_resolution_clock::now() - start_time).count();
//
//    // statistics
//    std::cout << "- Time cost: " << time_cost << "ms" << std::endl;
//
//    start_time = std::chrono::high_resolution_clock::now();
//    appr_alg->resizeIndex(X.n);
//#pragma omp parallel for schedule(dynamic, 144)
//    for (int i = 500000; i < X.n; i++) {
//        appr_alg->addPoint(X.data + i * X.d, i);
//    }
//    cout<<"finished next half"<<std::endl;
//
//    time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(
//            std::chrono::high_resolution_clock::now() - start_time).count();
//
//    // statistics
//    std::cout << "- Time cost: " << time_cost << "ms" << std::endl;

    auto l2space = new hnswlib::L2Space(X.d);
    auto appr_alg = new hnswlib::HierarchicalNSWStatic<float>(l2space, X.n, 16, 200);
    auto start_time = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 144)
    for (int i = 0; i < X.n; i++) {
        appr_alg->addPoint(X.data + i * X.d, i);
    }
    cout<<"finished half"<<std::endl;
    auto time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time).count();

    // statistics
    std::cout << "- Time cost: " << time_cost << "ms" << std::endl;

    return 0;
}
