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
    sprintf(index_path, "%s%s_opt.hnsw", source, dataset);
    sprintf(label_path, "%s%s_base_%s.txt", source, dataset, label);
    Matrix<float> X(data_path);
    hnswlib::HierarchicalNSWStatic<float>::static_base_data_ = (char *) X.data;
    IndexLabelOpt hnsw_opt(X.n, X.d, X.data);
    hnsw_opt.load_base_label_bitmap(label_path);
    hnsw_opt.build_optimal_index(X);
    hnsw_opt.save_optimal_index(index_path);
    return 0;
}
