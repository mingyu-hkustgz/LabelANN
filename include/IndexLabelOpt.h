//
// Created by mingyu on 25-2-15.
//

#include "utils.h"
#include "matrix.h"
#include "hnswlib/hnswalg-static.h"
#include "hnswlib/hnswlib.h"
#include <mutex>

template<typename dist_t> char *hnswlib::HierarchicalNSWStatic<dist_t>::static_base_data_ = NULL;
#define HNSW_M 16
#define HNSW_efConstruction 200
#define POWER_BOUND 10
#define INDEX_BOUND 4096
using namespace std;

class IndexLabelOpt {
public:
    IndexLabelOpt() {}

    IndexLabelOpt(unsigned num, unsigned dim) {
        N = num;
        D = dim;
    }

    ResultQueue bruteforce_range_search(std::vector<size_t> &id, const float *query, unsigned K) {
        ResultQueue res;
        for (auto u:id) {
            float dist = sqr_dist(data_ + u * D, query, D);
            if (res.size() < K)
                res.emplace(dist, u);
            else if (dist < res.top().first) {
                res.pop();
                res.emplace(dist, u);
            }
        }
        return res;
    }

    std::priority_queue<std::pair<float, hnswlib::labeltype> >
    naive_search(const float *query, unsigned K, unsigned nprobs) {
        auto appr_alg = appr_alg_list[0];
        return appr_alg->searchKnn(query, K, nprobs);
    }


    std::priority_queue<std::pair<float, hnswlib::labeltype> >
    contain_search(const float *query, unsigned K, unsigned nprobs, uint64_t bitmap) {
        auto appr_alg = appr_alg_list[bitmap];
        return appr_alg->searchKnn(query, K, nprobs);
    }

    std::pair<ANNS::IdxType, float> *
    contain_parallel_search(const float *query, unsigned K, unsigned nprobs, int nthread) {
        omp_set_num_threads(nthread);
        auto results = new std::pair<ANNS::IdxType, float>[query_bitmap.size() * K];
#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < query_bitmap.size(); i++) {
            auto bitmap = query_bitmap[i];

            if(bitmap_list[bitmap].size() >= INDEX_BOUND){
                auto appr_alg = appr_alg_list[bitmap];
                auto hnsw_result = appr_alg->searchKnn(query + i * D, K, nprobs);
                unsigned back_tag = K;
                while (!hnsw_result.empty()) {
                    back_tag--;
                    results[i * K + back_tag].first = hnsw_result.top().second;
                    results[i * K + back_tag].second = hnsw_result.top().first;
                    hnsw_result.pop();
                }
            }
            else{
                auto brute_result = bruteforce_range_search(bitmap_list[bitmap],query + i * D, K);
                unsigned back_tag = K;
                while (!brute_result.empty()) {
                    back_tag--;
                    results[i * K + back_tag].first = brute_result.top().second;
                    results[i * K + back_tag].second = brute_result.top().first;
                    brute_result.pop();
                }
            }
        }
        return results;
    }


    void build_naive_index(Matrix<float> &X) {
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
        data_ = X.data;
    }

    void load_base_label_bitmap(const char *filename) {
        load_bitmap(filename, label_bitmap, N);
        std::cerr << "Base Label Load Finished" << std::endl;
    }

    void load_query_label_bitmap(const char *filename, unsigned query_num) {
        load_bitmap(filename, query_bitmap, query_num);
        std::cerr << "Query Label Load Finished" << std::endl;
    }


    void preprocess_optimal_vector() {
        power_points = 0;
        for (int i = 0; i < N; i++) {
            auto bit_map = label_bitmap[i];
            auto bit_size = _mm_popcnt_u64(bit_map);
            if (bit_size > POWER_BOUND) {
                bitmap_list[0].push_back(i);
                power_points ++;
                continue;
            }
            std::vector<short> bit_loc;
            short cnt = 0;
            while (bit_map) {
                if (bit_map & 1) bit_loc.push_back(cnt);
                cnt++;
                bit_map >>= 1;
            }
            for (uint64_t j = 0; j < (1 << bit_size); j++) {
                uint64_t point_bitmap = 0;
                uint64_t check_bit = j, check_loc = 0;
                while (check_bit) {
                    if (check_bit & 1) point_bitmap |= (1 << bit_loc[check_loc]);
                    check_loc++;
                    check_bit >>= 1;
                }
                bitmap_list[point_bitmap].push_back(i);
            }
            power_points += (1<<bit_size);
        }
    }

    void build_optimal_index(Matrix<float> &X) {
        preprocess_optimal_vector();
        uint64_t cumulate_points = 0;
        std::cout<<"All Points:: "<<power_points<<std::endl;
        for (const auto &u: bitmap_list) {
            auto points_num = u.second.size();
            if (points_num < INDEX_BOUND){
                cumulate_points += points_num;
                std::cout << "\r Processing::" << (100.0 * cumulate_points) / power_points<< "%" << std::flush;
                continue;
            }
            auto l2space = new hnswlib::L2Space(D);
            auto appr_alg = new hnswlib::HierarchicalNSWStatic<float>(l2space, points_num, HNSW_M, HNSW_efConstruction);
            std::cout << "\r Processing:: " << (100.0 * cumulate_points) / power_points<< "%" << std::flush;
#pragma omp parallel for schedule(dynamic, 144)
            for (int i = 0; i < points_num; i++) {
                appr_alg->addPoint(X.data + (size_t)u.second[i] * D, u.second[i]);
            }
            cumulate_points += points_num;
            appr_alg_list[u.first] = appr_alg;
        }
        data_ = X.data;
    }

    void save_single_static_index(hnswlib::HierarchicalNSWStatic<float> *&appr_alg, std::ofstream &fout) {
        fout.write((char *) &appr_alg->enterpoint_node_, sizeof(unsigned int));
        fout.write((char *) &appr_alg->maxlevel_, sizeof(unsigned int));
        for (size_t j = 0; j < appr_alg->cur_element_count; j++) {
            unsigned int linkListSize = appr_alg->element_levels_[j] > 0 ? appr_alg->size_links_per_element_ *
                                                                           appr_alg->element_levels_[j] : 0;
            fout.write((char *) &linkListSize, sizeof(unsigned));
            if (linkListSize)
                fout.write((char *) appr_alg->linkLists_[j], linkListSize);
        }
        fout.write((char *) appr_alg->data_level0_memory_,
                   appr_alg->cur_element_count * appr_alg->size_data_per_element_);
    }

    void
    load_single_static_index(hnswlib::HierarchicalNSWStatic<float> *&appr_alg, std::ifstream &fin, uint64_t bitmap) {
        auto l2space = new hnswlib::L2Space(D);
        auto hnsw_size = bitmap_list[bitmap].size();
        appr_alg = new hnswlib::HierarchicalNSWStatic<float>(l2space, hnsw_size, HNSW_M, HNSW_efConstruction);
        fin.read((char *) &appr_alg->enterpoint_node_, sizeof(unsigned int));
        fin.read((char *) &appr_alg->maxlevel_, sizeof(unsigned int));
        appr_alg->cur_element_count = hnsw_size;
        for (size_t j = 0; j < hnsw_size; j++) {
            unsigned int linkListSize;
            fin.read((char *) &linkListSize, sizeof(unsigned));
            if (linkListSize == 0) {
                appr_alg->element_levels_[j] = 0;
                appr_alg->linkLists_[j] = nullptr;
            } else {
                appr_alg->element_levels_[j] = linkListSize / appr_alg->size_links_per_element_;
                appr_alg->linkLists_[j] = (char *) malloc(linkListSize);
                if (appr_alg->linkLists_[j] == nullptr) {
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                }
                fin.read(appr_alg->linkLists_[j], linkListSize);
            }
        }
        fin.read((char *) appr_alg->data_level0_memory_,
                 appr_alg->cur_element_count * appr_alg->size_data_per_element_);
    }

    void save_optimal_index(char *filename) {
        std::ofstream fout(filename, std::ios::binary);
        unsigned map_size = bitmap_list.size();
        fout.write((char *) &map_size, sizeof(unsigned));
        for (auto u: bitmap_list) {
            unsigned size = u.second.size();
            fout.write((char *) &u.first, sizeof(uint64_t));
            fout.write((char *) &size, sizeof(unsigned));
            fout.write((char *) u.second.data(), sizeof(size_t) * (size_t) size);
            if (appr_alg_list[u.first] == nullptr || size < INDEX_BOUND) continue;
            save_single_static_index(appr_alg_list[u.first], fout);
        }
    }

    void load_optimal_index(char *filename) {
        std::ifstream fin(filename, std::ios::binary);
        unsigned map_size, size;
        uint64_t cumulate_points = 0;
        fin.read((char *) &map_size, sizeof(unsigned));
        for (int i = 0; i < map_size; i++) {
            uint64_t bitmap;
            fin.read((char *) &bitmap, sizeof(uint64_t));
            fin.read((char *) &size, sizeof(unsigned));
            cumulate_points += size;
            bitmap_list[bitmap].resize(size);
            fin.read((char *) bitmap_list[bitmap].data(), sizeof(size_t) * (size_t) size);
            if (size >= INDEX_BOUND)
                load_single_static_index(appr_alg_list[bitmap], fin, bitmap);
        }
        power_points = cumulate_points;
        std::cerr<<"All Points:: "<<cumulate_points<<std::endl;
    }


    unsigned N, D;
    uint64_t power_points;
    std::vector<uint64_t> label_bitmap, query_bitmap;
    std::unordered_map<uint64_t, std::vector<size_t> > bitmap_list;
    std::unordered_map<uint64_t, hnswlib::HierarchicalNSWStatic<float> *> appr_alg_list;
    std::vector<std::pair<hnswlib::labeltype, hnswlib::labeltype> > index_range_list;
    float *data_;
};
