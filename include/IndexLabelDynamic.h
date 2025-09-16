//
// Created by mingyu on 25-2-15.
//

#include "utils.h"
#include "matrix.h"
#include "hnswlib/hnswalg-static.h"
#include "hnswlib/hnswlib.h"
#include <mutex>
#include "IndexLabelElastic.h"
using namespace std;


class IndexLabelDynamic: public IndexLabelElastic {
public:
    IndexLabelDynamic() {}

    IndexLabelDynamic(unsigned num, unsigned dim) {
        N = num;
        D = dim;
    }

    IndexLabelDynamic(unsigned num, unsigned dim, float *base_data) {
        N = num;
        D = dim;
        data_ = base_data;
    }

    void update_index(unsigned num, const char*filename){
        N += num;
        label_bitmap.clear();
        bitmap_list.clear();
        indexed_points = 0;
        load_base_label_bitmap(filename);
        preprocess_optimal_vector();
    }


    void reduce_index(){
        uint64_t count = 0, reduced_size = 0;
        std::unordered_map<uint64_t, bool> index_map;
        for(auto bitmap:selected_bitmap) index_map[bitmap] = true;
        for (auto it = appr_alg_list.begin(); it != appr_alg_list.end();) {
            if (!index_map[it->first] && it->second != nullptr) {
                reduced_size += it->second->cur_element_count;
                count++;
                delete it->second; // 释放指针指向的内存
                it = appr_alg_list.erase(it);
            } else {
                ++it;
            }
        }
        indexed_points -= reduced_size;
        std::cout<<"Reduce Index Number:: "<<count<<" Reduce Index Size:: "<<reduced_size<<std::endl;
    }

    uint64_t update_best_fit_schedule() {
        uint64_t total_cost = 0, total_points=0;
        uint64_t set_covered_count = 0;
        std::unordered_map<uint64_t, bool> set_check, element_check;
        std::unordered_map<uint64_t, double> set_benefit, set_cover_num;
        std::unordered_map<uint64_t, uint64_t> set_cost;
        std::unordered_map<uint64_t, std::vector<uint64_t>> element_to_sets;
        std::priority_queue<std::pair<double, uint64_t> > queue;
        for (auto &item: bitmap_list) {
            if (item.second.size() < INDEX_ELASIIC_BOUND) continue;
            double benefit = 0, cover_num = 0;
            for (auto u: cover_set_[item.first]) {
                cover_num += (double) bitmap_list[u].size();
                if(appr_alg_list[u]!= nullptr) cover_num -= (double) appr_alg_list[u]->cur_element_count;
                element_to_sets[u].push_back(item.first);
            }
            auto index_cost = item.second.size();
            if(appr_alg_list[item.first]!= nullptr) index_cost -= appr_alg_list[item.first]->cur_element_count;
            if(index_cost == 0) index_cost++;
            benefit = (double) cover_num / (double) index_cost;
            set_benefit[item.first] = benefit + 1.00/(double) index_cost;
            set_cost[item.first] = index_cost;
            set_cover_num[item.first] = cover_num;
            if (item.first == 0){
                set_benefit[item.first] = MAXFLOAT;
                benefit = MAXFLOAT;
            }
            queue.emplace(benefit, item.first);
        }

        while (set_covered_count < index_set_count) {
            auto top_set = queue.top();
            auto bitmap = top_set.second;
            auto benefit = top_set.first;
            queue.pop();
            if (std::abs(set_benefit[bitmap] - benefit) < 1e-6) {
                selected_bitmap.push_back(bitmap);
                total_cost += set_cost[bitmap];
                total_points += bitmap_list[bitmap].size();
                set_check[bitmap] = true;
                set_covered_count += cover_set_[bitmap].size();
                for (auto element: cover_set_[bitmap]) {
                    if (element_check[element]) continue;
                    element_check[element] = true;
                    for (auto influence_set: element_to_sets[element]) {
                        if (set_check[influence_set]) continue;
                        set_cover_num[influence_set] -= (double) set_cost[element];
                        set_benefit[influence_set] =
                                set_cover_num[influence_set] / (double) set_cost[influence_set];
                        set_benefit[influence_set] += 1.00 / (double) set_cost[influence_set];
                    }
                }
            } else {
                queue.emplace(set_benefit[top_set.second], top_set.second);
            }
        }
        std::cout<<"Total Cost: "<<total_cost<<" with Elastic Factor: "<<elastic_factor_<<std::endl;
        return total_points;
    }

    void binary_search_dynamic_factor(uint64_t cost_bound){
        std::cout<<"Begin Binary Dynamic Search with Cost Bound "<<cost_bound<<std::endl;
        float lower_bound= 0.00, upper_bound = 1.0, final_elastic = 0.0;
        std::vector<uint64_t> final_ans;
        while(lower_bound + 1e-3 < upper_bound){
            float mid = (lower_bound + upper_bound)/2.0;
            std::cout<<"Current Dynamic Elastic Factor "<<elastic_factor_<<std::endl;
            elastic_factor_ = mid;
            selected_bitmap.clear();
            preprocess_cover_relationship();
            uint64_t cost = update_best_fit_schedule();
            if(cost < cost_bound){
                lower_bound = mid;
                final_elastic = mid;
                final_ans = selected_bitmap;
                std::cout<<"Find Solution with Cost:: "<<cost<<std::endl;
            }
            else{
                upper_bound = mid;
            }
        }
        selected_bitmap = final_ans;
        elastic_factor_ = final_elastic;
    }

    void build_elastic_index(Matrix<float> &X) override {
        preprocess_optimal_vector();
        if(elastic_factor_ < 1.0){
            preprocess_cover_relationship();
            best_fit_schedule();
        }
        else
            binary_search_elastic_factor((uint64_t) (N * (double)elastic_factor_));
        find_cover_father();
        uint64_t cumulate_points = 0;
        std::cout << "All Points:: " << indexed_points << std::endl;
        for (const auto &bitmap: selected_bitmap) {
            auto vec_list = bitmap_list[bitmap];
            auto points_num = vec_list.size();
            if (points_num < INDEX_ELASIIC_BOUND) {
                cumulate_points += points_num;
                std::cout<<0<<" xx "<<points_num<<std::endl;
                std::cout << "\r Processing::" << (100.0 * cumulate_points) / indexed_points << "%" << std::flush;
                continue;
            }
            auto l2space = new hnswlib::L2Space(D);
            auto appr_alg = new hnswlib::HierarchicalNSWStatic<float>(l2space, points_num<<1, HNSW_ELASTIC_M,
                                                                      HNSW_ELASIIC_efConstruction);
            std::cout << "\r Processing:: " << (100.0 * cumulate_points) / indexed_points << "%" << std::flush;
#pragma omp parallel for schedule(dynamic, 144)
            for (int i = 0; i < points_num; i++) {
                appr_alg->addPoint(X.data + (size_t) vec_list[i] * (size_t) D, vec_list[i]);
            }
            cumulate_points += points_num;
            appr_alg_list[bitmap] = appr_alg;
        }
        data_ = X.data;
        std::cout << "\r Processing:: " << (100.0 * cumulate_points) / indexed_points << "%" << std::flush;
    }

    void incremental_elastic_index(){
        if(elastic_factor_ < 1.0){
            preprocess_cover_relationship();
            selected_bitmap.clear();
            update_best_fit_schedule();
        }
        else{
            binary_search_dynamic_factor((uint64_t) (N * (double)elastic_factor_));
        }
        reduce_index();
        find_cover_father();
        uint64_t cumulate_points = 0;
        std::cout << "All Points:: " << indexed_points << std::endl;
        for (const auto &bitmap: selected_bitmap) {
            auto vec_list = bitmap_list[bitmap];
            auto points_num = vec_list.size();
            if (points_num < INDEX_ELASIIC_BOUND) {
                cumulate_points += points_num;
                std::cout << "\r Processing::" << (100.0 * cumulate_points) / indexed_points << "%" << std::flush;
                continue;
            }
            hnswlib::HierarchicalNSWStatic<float> * appr_alg;
            if(appr_alg_list[bitmap] == nullptr) {
                auto l2space = new hnswlib::L2Space(D);
                appr_alg = new hnswlib::HierarchicalNSWStatic<float>(l2space, points_num, HNSW_ELASTIC_M, HNSW_ELASIIC_efConstruction);
                cumulate_points += points_num;
            }
            else{
                appr_alg = appr_alg_list[bitmap];
                if(appr_alg->max_elements_ < points_num) appr_alg->resizeIndex(points_num);
                cumulate_points += points_num;
            }
            std::cout << "\r Processing:: " << (100.0 * cumulate_points) / indexed_points << "%" << std::flush;
            uint64_t pre_size = appr_alg->cur_element_count;
#pragma omp parallel for schedule(dynamic, 144)
            for(uint64_t i = pre_size; i < points_num; i++){
                appr_alg->addPoint(data_+ (size_t) vec_list[i] * (size_t) D, vec_list[i]);
            }
            appr_alg_list[bitmap] = appr_alg;
        }
        std::cout << "\r Processing:: " << (100.0 * cumulate_points) / indexed_points << "%" << std::flush;
    }

    std::pair<ANNS::IdxType, float> *
    contain_search(const float *query, unsigned K, unsigned nprobs) {
        auto results = new std::pair<ANNS::IdxType, float>[query_bitmap.size() * K];
        for (int i = 0; i < query_bitmap.size(); i++) {
            auto bitmap = query_bitmap[i];
            if (bitmap_list[bitmap].size() >= INDEX_ELASIIC_BOUND) {
                if(fa_[bitmap]!=bitmap){
                    auto appr_alg = appr_alg_list[fa_[bitmap]];
                    ContainLabelFilter contain_filter(bitmap);
                    auto hnsw_result = appr_alg->searchKnn(query + i * D, K, nprobs, &contain_filter);
                    unsigned back_tag = K;
                    while (!hnsw_result.empty()) {
                        back_tag--;
                        results[i * K + back_tag].first = hnsw_result.top().second;
                        results[i * K + back_tag].second = hnsw_result.top().first;
                        hnsw_result.pop();
                    }
                }
                else{
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
            } else {
                auto brute_result = bruteforce_range_search(bitmap_list[bitmap], query + i * D, K);
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

};

