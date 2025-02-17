//
// Created by mingyu on 25-2-15.
//
#ifndef INDEX_LABEL_H
#define INDEX_LABEL_H

#include "utils.h"
#include "matrix.h"
#include "hnswlib/hnswalg-static.h"
#include "hnswlib/hnswlib.h"

template<typename dist_t> char *hnswlib::HierarchicalNSWStatic<dist_t>::static_base_data_ = NULL;
#define HNSW_M 16
#define HNSW_efConstruction 200
using namespace std;

class IndexLabelOpt {
public:
    IndexLabelOpt() {}

    IndexLabelOpt(unsigned dim) {
        D = dim;
    }

    IndexLabelOpt(unsigned num, unsigned dim) {
        N = num;
        D = dim;
    }

    std::priority_queue<std::pair<float, hnswlib::labeltype> >
    naive_search(const float *query, unsigned K, unsigned nprobs) {
        auto appr_alg = appr_alg_list[0];
        appr_alg->setEf(nprobs);
        return appr_alg->searchKnn(query, K);
    }

    void build_index(Matrix<float> &X) {
        std::cout << "build index" << std::endl;
        unsigned check_tag = 0, report = 50000;
        auto l2space = new hnswlib::L2Space(D);
        auto appr_alg = new hnswlib::HierarchicalNSWStatic<float>(l2space, N, HNSW_M, HNSW_efConstruction);
#pragma omp parallel for schedule(dynamic, 144)
        for (int i = 0; i < N; i++) {
            appr_alg->addPoint(X.data + i * D, i);
#pragma omp critical
            {
                check_tag++;
                if (check_tag % report == 0) {
                    std::cerr << "Processing - " << check_tag << " / " << N << std::endl;
                }
            }
        }
        appr_alg_list[0] = appr_alg;
    }

    void load_base_label(const char *filename) {
        if(!isFileExists_ifstream(filename)){
            std::cerr<<"Label File Not Exists"<<std::endl;
            return;
        }

        std::ifstream fin(filename);
        std::string line;
        label_bitmap.reserve(N);
        while (std::getline(fin, line)) {
            uint64_t bitmap = 0;
            std::istringstream iss(line);
            std::string token;
            std::vector<std::string> tokens;
            while (std::getline(iss, token, ',')) {
                tokens.push_back(token);
            }
            for (const auto &t: tokens) {
                bitmap |= (1 << std::atoi(t.c_str()));
            }
            label_bitmap.push_back(bitmap);
        }
        std::cerr<<"Base Label Load Finished"<<std::endl;
        for(int i=0;i<10;i++) std::cout<<label_bitmap[i]<<std::endl;
    }


    unsigned N, D;
    std::vector<uint64_t> label_bitmap;
    std::unordered_map<uint64_t, hnswlib::HierarchicalNSWStatic<float> *> appr_alg_list;
    std::vector<std::pair<hnswlib::labeltype, hnswlib::labeltype> > index_range_list;
};


#endif //INDEX_LABEL_H
