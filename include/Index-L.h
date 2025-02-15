//
// Created by mingyu on 25-2-15.
//
#ifndef INDEX_LABEL_H
#define INDEX_LABEL_H
#include "utils.h"
#include "matrix.h"
#include "hnswlib/hnswalg.h"
#include "hnswlib/hnswlib.h"

#define HNSW_M 16
#define HNSW_efConstruction 200
using namespace std;

class IndexLabel {
public:
    IndexLabel() {}

    IndexLabel(unsigned dim){
        D = dim ;
    }

    IndexLabel(unsigned num, unsigned dim){
        N = num;
        D = dim;
    }

    std::priority_queue<std::pair<float, hnswlib::labeltype> >
    naive_search(const float *query, unsigned K, unsigned nprobs) {
        auto appr_alg = appr_alg_list[0];
        appr_alg->setEf(nprobs);
        return appr_alg->searchKnn(query, K);
    }

    void build_index(Matrix<float> &X){
        std::cout<<"build index"<<std::endl;
        unsigned check_tag = 0, report = 50000;
        hnswlib::L2Space l2space(D);
        auto appr_alg = new hnswlib::HierarchicalNSW<float>(&l2space, N, HNSW_M, HNSW_efConstruction);
#pragma omp parallel for schedule(dynamic, 144)
        for (hnswlib::labeltype i = 0;i < N; i++) {
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

    unsigned N, D;
    std::unordered_map<uint64_t, hnswlib::HierarchicalNSW<float> *> appr_alg_list;
    std::vector<std::pair<hnswlib::labeltype, hnswlib::labeltype> > index_range_list;
};



#endif //INDEX_LABEL_H
