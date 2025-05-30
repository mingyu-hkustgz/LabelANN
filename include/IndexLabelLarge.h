//
// Created by mingyu on 25-2-15.
//

#include "utils.h"
#include "matrix.h"
#include "hnswlib/hnswalg-static.h"
#include "hnswlib/hnswlib.h"
#include <mutex>
#include <boost/dynamic_bitset.hpp>

template<typename dist_t> char *hnswlib::HierarchicalNSWStatic<dist_t>::static_base_data_ = NULL;
#define HNSW_ELASTIC_M 16
#define HNSW_ELASIIC_efConstruction 200
#define POWER_ELASIIC_BOUND 10
#define INDEX_ELASIIC_BOUND 4096
using namespace std;
using namespace boost;

class ContainLabelFilter : public hnswlib::BaseFilterFunctor {
public:
    dynamic_bitset<> query_bitmap;
    static dynamic_bitset<> *label_bit_map_;
    unsigned query_bits = 0;

    ContainLabelFilter(dynamic_bitset<> query_bit) {
        query_bitmap = query_bit;
        query_bits = query_bit.count();
    }

    bool operator()(hnswlib::labeltype id) override {
        return (query_bitmap & label_bit_map_[id]).count() >= query_bits;
    }
};
class InterLabelFilter : public hnswlib::BaseFilterFunctor {
public:
    dynamic_bitset<> query_bitmap;
    static dynamic_bitset<> *label_bit_map_;

    InterLabelFilter(dynamic_bitset<> query_bit) {
        query_bitmap = query_bit;
    }

    bool operator()(hnswlib::labeltype id) override {
        return (query_bitmap & label_bit_map_[id]).count() > 0;
    }
};

dynamic_bitset<> *ContainLabelFilter::label_bit_map_ = nullptr;
dynamic_bitset<> *InterLabelFilter::label_bit_map_ = nullptr;

void load_large_bitmap(const char *filename, std::vector<dynamic_bitset<>> &label_bitmap, unsigned points_num) {
    if (!isFileExists_ifstream(filename)) {
        std::cerr << "Label File Not Exists" << std::endl;
        return;
    }
    unsigned cumulate_points = 0;
    std::ifstream fin(filename);
    std::string line;
    label_bitmap.reserve(points_num);
    unsigned max_bits = 0;
    while (std::getline(fin, line)) {
        cumulate_points ++;
        dynamic_bitset<> bitmap(512, 0);
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(iss, token, ',')) {
            tokens.push_back(token);
        }
        for (const auto &t: tokens) {
            bitmap.set(std::atoi(t.c_str()) - 1, true);
        }
        label_bitmap.push_back(bitmap);
        if (bitmap.count() > max_bits) max_bits = bitmap.count();
        if(cumulate_points==points_num) break;
    }
    std::cerr << "Max bits:" << max_bits << std::endl;
}

class IndexLabelElastic {
public:
    IndexLabelElastic() {}

    IndexLabelElastic(unsigned num, unsigned dim) {
        N = num;
        D = dim;
    }

    IndexLabelElastic(unsigned num, unsigned dim, float *base_data) {
        N = num;
        D = dim;
        data_ = base_data;
    }

    void set_elastic_factor(float factor) {
        elastic_factor_ = factor;
    }

    ResultQueue bruteforce_range_search(std::vector<size_t> &id, const float *query, unsigned K) {
        ResultQueue res;
        for (auto u: id) {
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

    std::pair<ANNS::IdxType, float> *
    contain_parallel_search(const float *query, unsigned K, unsigned nprobs, int nthread) {
//        omp_set_num_threads(nthread);
        auto results = new std::pair<ANNS::IdxType, float>[query_bitmap.size() * K];
//#pragma omp parallel for schedule(dynamic, nthread)
        for (int i = 0; i < query_bitmap.size(); i++) {
            auto bitmap = query_bitmap[i];
            if (bitmap_list[bitmap].size() >= INDEX_ELASIIC_BOUND) {
                if(fa_[bitmap]!=bitmap){
                    if(fa_[bitmap].count()==0) fa_[bitmap].resize(512);
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

    void load_base_label_bitmap(const char *filename) {
        load_large_bitmap(filename, label_bitmap, N);
        ContainLabelFilter::label_bit_map_ = label_bitmap.data();
        InterLabelFilter::label_bit_map_ = label_bitmap.data();
        std::cout << "Base Label Load Finished" << std::endl;
    }

    void load_query_label_bitmap(const char *filename, unsigned query_num) {
        load_large_bitmap(filename, query_bitmap, query_num);
        std::cout << "Query Label Load Finished" << std::endl;
    }

    void preprocess_optimal_vector() {
        power_points = 0;
        for (int i = 0; i < N; i++) {
            auto bit_map = label_bitmap[i];
            auto bit_size = bit_map.count();
            std::vector<short> bit_loc;
            for (short j = bit_map.find_first(); j != boost::dynamic_bitset<>::npos;) {
                bit_loc.push_back(j);
                j = bit_map.find_next(j);
            }
            for (uint64_t j = 0; j < (1 << bit_size); j++) {
                dynamic_bitset<> point_bitmap(512, 0);
                uint64_t check_bit = j, check_loc = 0;
                while (check_bit) {
                    if (check_bit & 1) point_bitmap.set(bit_loc[check_loc], true);
                    check_loc++;
                    check_bit >>= 1;
                }
                bitmap_list[point_bitmap].push_back(i);
            }
            power_points += (1 << bit_size);
        }
        std::cout << "Power points " << power_points << std::endl;
    }

    void preprocess_cover_relationship() {
        cover_set_.clear();
        index_set_count = 0;
        for (auto &item: bitmap_list) {
            if (item.second.size() < INDEX_ELASIIC_BOUND) continue;
            index_set_count++;
            auto bit_map = item.first;
            auto bit_size = bit_map.count();
            auto index_size = item.second.size();
            std::vector<short> bit_loc;
            short cnt = 0;
            for (short j = bit_map.find_first(); j != boost::dynamic_bitset<>::npos;) {
                bit_loc.push_back(j);
                j = bit_map.find_next(j);
            }
            for (uint64_t j = 0; j < (1 << bit_size); j++) {
                dynamic_bitset<> point_bitmap(512, 0);
                uint64_t check_bit = j, check_loc = 0;
                while (check_bit) {
                    if (check_bit & 1) point_bitmap.set(bit_loc[check_loc], true);
                    check_loc++;
                    check_bit >>= 1;
                }
                auto cover_size = bitmap_list[point_bitmap].size();
                if ((double) index_size / (double) cover_size > elastic_factor_)
                    cover_set_[point_bitmap].push_back(item.first);
            }
        }

    }

    uint64_t best_fit_schedule() {
        uint64_t total_cost = 0;
        uint64_t set_covered_count = 0;
        std::unordered_map<dynamic_bitset<>, bool> set_check, element_check;
        std::unordered_map<dynamic_bitset<>, double> set_benefit, set_cover_num;
        std::unordered_map<dynamic_bitset<>, std::vector<dynamic_bitset<>>> element_to_sets;
        std::priority_queue<std::pair<double, dynamic_bitset<>> > queue;
        for (auto &item: bitmap_list) {
            if (item.second.size() < INDEX_ELASIIC_BOUND) continue;
            double benefit = 0, cover_num = 0;
            for (auto u: cover_set_[item.first]) {
                cover_num += (double) bitmap_list[u].size();
                element_to_sets[u].push_back(item.first);
            }
            benefit = (double) cover_num / (double) item.second.size();
            set_benefit[item.first] = benefit + 1.00/(double)item.second.size();
            set_cover_num[item.first] = cover_num;
            if (item.first.count()==0){
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
                total_cost += bitmap_list[bitmap].size();
                set_check[bitmap] = true;
                set_covered_count += cover_set_[bitmap].size();
                for (auto element: cover_set_[bitmap]) {
                    if (element_check[element]) continue;
                    element_check[element] = true;
                    for (auto influence_set: element_to_sets[element]) {
                        if (set_check[influence_set]) continue;
                        set_cover_num[influence_set] -= (double) bitmap_list[element].size();
                        set_benefit[influence_set] =
                                set_cover_num[influence_set] / (double) bitmap_list[influence_set].size();
                        set_benefit[influence_set] += 1.00/(double)bitmap_list[influence_set].size();
                    }
                }
            } else {
                queue.emplace(set_benefit[top_set.second], top_set.second);
            }
        }
        std::cout<<"Total Cost: "<<total_cost<<" with Elastic Factor: "<<elastic_factor_<<std::endl;
        return total_cost;
    }

    void find_cover_father() {
        std::unordered_map<dynamic_bitset<>, double> cover_ratio;
        for (auto father: selected_bitmap) {
            indexed_points += bitmap_list[father].size();
            for (auto son: cover_set_[father]) {
                double element_elastic_factor =
                        (double) bitmap_list[son].size() / (double) bitmap_list[father].size();
                if (element_elastic_factor > cover_ratio[son]) {
                    cover_ratio[son] = element_elastic_factor;
                    fa_[son] = father;
                }
            }
        }
    }

    void binary_search_elastic_factor(uint64_t cost_bound){
        std::cout<<"Begin Binary Search with Cost Bound "<<cost_bound<<std::endl;
        float lower_bound= 0.00, upper_bound = 1.0, final_elastic = 0.0;
        std::vector<dynamic_bitset<>> final_ans;
        while(lower_bound + 1e-3 < upper_bound){
            float mid = (lower_bound + upper_bound)/2.0;
            std::cout<<"Current Elastic Factor "<<elastic_factor_<<std::endl;
            elastic_factor_ = mid;
            selected_bitmap.clear();
            preprocess_cover_relationship();
            uint64_t cost = best_fit_schedule();
            if(cost < cost_bound){
                lower_bound = mid;
                final_elastic = mid;
                final_ans = selected_bitmap;
            }
            else{
                upper_bound = mid;
            }
        }
        selected_bitmap = final_ans;
        elastic_factor_ = final_elastic;
    }


    void build_elastic_index(Matrix<float> &X) {
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
                std::cout << "\r Processing::" << (100.0 * cumulate_points) / indexed_points << "%" << std::flush;
                continue;
            }
            auto l2space = new hnswlib::L2Space(D);
            auto appr_alg = new hnswlib::HierarchicalNSWStatic<float>(l2space, points_num, HNSW_ELASTIC_M,
                                                                      HNSW_ELASIIC_efConstruction);
            std::cout << "\r Processing:: " << (100.0 * cumulate_points) / indexed_points << "%" << std::flush;
#pragma omp parallel for schedule(dynamic, 144)
            for (int i = 0; i < points_num; i++) {
#ifndef ID_COMPACT
                appr_alg->addPoint(X.data + (size_t) vec_list[i] * D, vec_list[i]);
#else
                appr_alg->addPoint(X.data + (size_t) vec_list[i] * D, (vec_list[i]<<32) + label_bitmap[vec_list[i]]);
#endif
            }
            cumulate_points += points_num;
            appr_alg_list[bitmap] = appr_alg;
        }
        data_ = X.data;
        std::cout << "\r Processing:: " << (100.0 * cumulate_points) / indexed_points << "%" << std::flush;
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
    load_single_static_index(hnswlib::HierarchicalNSWStatic<float> *&appr_alg, std::ifstream &fin, dynamic_bitset<> bitmap) {
        auto l2space = new hnswlib::L2Space(D);
        auto hnsw_size = bitmap_list[bitmap].size();
        appr_alg = new hnswlib::HierarchicalNSWStatic<float>(l2space, hnsw_size, HNSW_ELASTIC_M,
                                                             HNSW_ELASIIC_efConstruction);
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

    void save_elastic_index(char *filename) {
        std::ofstream fout(filename, std::ios::binary);
        unsigned map_size = bitmap_list.size();
        fout.write((char *) &map_size, sizeof(unsigned));
        for (auto u: bitmap_list) {
            unsigned size = u.second.size();
            unsigned char bit_bytes[64] = {0};
            for (size_t j = 0; j < 512; ++j) {
                if (u.first.test(j)) {
                    bit_bytes[j / 8] |= (1 << (j % 8));
                }
            }
            fout.write((char *) bit_bytes, 64);
            fout.write((char *) &size, sizeof(unsigned));
            fout.write((char *) u.second.data(), sizeof(size_t) * (size_t) size);
            for (size_t j = 0; j < 512; ++j) {
                if (fa_[u.first].test(j)) {
                    bit_bytes[j / 8] |= (1 << (j % 8));
                }
            }
            fout.write((char *) bit_bytes, 64);
            if (appr_alg_list[u.first] == nullptr || size < INDEX_ELASIIC_BOUND) continue;
            if (fa_[u.first] == u.first)
                save_single_static_index(appr_alg_list[u.first], fout);
        }
    }

    void load_elastic_index(char *filename) {
        std::ifstream fin(filename, std::ios::binary);
        unsigned map_size, size;
        uint64_t cumulate_points = 0, index_points = 0;
        fin.read((char *) &map_size, sizeof(unsigned));
        for (int i = 0; i < map_size; i++) {
            dynamic_bitset<> bitmap, father;
            unsigned char bit_bytes[64] = {0};
            fin.read((char *) &bit_bytes, 64);
            for (size_t j = 0; j < 512; ++j) {
                if (bit_bytes[j/8] & (1<<(i%8))) {
                    bitmap.set(j, true);
                }
            }
            fin.read((char *) &size, sizeof(unsigned));
            cumulate_points += size;
            bitmap_list[bitmap].resize(size);
            fin.read((char *) bitmap_list[bitmap].data(), sizeof(size_t) * (size_t) size);
            fin.read((char *) &bit_bytes, 64);
            for (size_t j = 0; j < 512; ++j) {
                if (bit_bytes[j/8] & (1<<(i%8))) {
                    father.set(j, true);
                }
            }
            fa_[bitmap] = father;
            if (size >= INDEX_ELASIIC_BOUND && fa_[bitmap] == bitmap){
                load_single_static_index(appr_alg_list[bitmap], fin, bitmap);
                index_points += size;
            }
        }
        power_points = cumulate_points;
        std::cout << "All Points:: " << cumulate_points << std::endl;
        std::cout << "All Points:: " << index_points << std::endl;
    }


    unsigned N, D;
    float elastic_factor_ = 0.5;
    uint64_t power_points = 0, indexed_points = 0, index_set_count = 0;
    std::vector<dynamic_bitset<>> label_bitmap, query_bitmap, selected_bitmap;
    std::unordered_map<dynamic_bitset<>, std::vector<dynamic_bitset<>> > cover_set_;
    std::unordered_map<dynamic_bitset<>, std::vector<uint64_t> > bitmap_list;
    std::unordered_map<dynamic_bitset<>, dynamic_bitset<>> fa_;
    std::unordered_map<dynamic_bitset<>, hnswlib::HierarchicalNSWStatic<float> *> appr_alg_list;
    float *data_;
};

