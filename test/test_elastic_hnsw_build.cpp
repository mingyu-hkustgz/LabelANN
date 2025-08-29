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
    int iarg = 0;
    float elastic_factor = 0.5;
    opterr = 1; //getopt error message (off: 0)

    char dataset[256] = "";
    char source[256] = "";
    char label[256] = "";
    char data_path[256] = "";
    char index_path[256] = "";
    char label_path[256] = "";
    char logger_path[256] = "";
    char file_type[256]= "fvecs";
    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:s:l:e:f:", longopts, &ind);
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
            case 'f':
                if (optarg) {
                    strcpy(file_type, optarg);
                }
                break;
        }
    }
    sprintf(data_path, "%s%s_base.%s", source, dataset, file_type);
#ifndef ID_COMPACT
    sprintf(index_path, "%s%s_elastic_%.2f.hnsw", source, dataset, elastic_factor);
#else
    sprintf(index_path, "%s%s_elastic_%.2f_compact.hnsw", source, dataset, elastic_factor);
#endif
    sprintf(label_path, "%s%s_base_%s.txt", source, dataset, label);
    sprintf(logger_path, "./results/%s-%.2f-%s.log", dataset, elastic_factor,label);
    Matrix<float> X(data_path);
    hnswlib::HierarchicalNSWStatic<float>::static_base_data_ = (char *) X.data;
    IndexLabelElastic hnsw_elastic(X.n, X.d, X.data);
    hnsw_elastic.load_base_label_bitmap(label_path);
    hnsw_elastic.set_elastic_factor(elastic_factor);
    hnsw_elastic.build_elastic_index(X);
    hnsw_elastic.save_elastic_index(index_path);
    hnsw_elastic.save_log(logger_path);
    return 0;
}
