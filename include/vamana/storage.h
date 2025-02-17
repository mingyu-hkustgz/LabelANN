#ifndef ANNS_STORAGE_H
#define ANNS_STORAGE_H

#include <limits>
#include <string>
#include <vector>
#include <memory>
#include <xmmintrin.h>
#include <immintrin.h>
#include "config.h"
#include "distance.h"

namespace ANNS {

    // interface for storage
    class IStorage {
    public:
        virtual ~IStorage() = default;

        // I/O
        virtual void load_from_file(const std::string &bin_file, const std::string &label_file,
                                    IdxType max_num_points = std::numeric_limits<IdxType>::max()) = 0;

        virtual void write_to_file(const std::string &bin_file, const std::string &label_file) = 0;

        // reorder the vector data
        virtual void reorder_data(const std::vector<IdxType> &new_to_old_ids) = 0;

        // get statistics
        virtual DataType get_data_type() const = 0;

        virtual IdxType get_num_points() const = 0;

        virtual IdxType get_dim() const = 0;

        // get data
        virtual std::vector<LabelType> *get_offseted_label_sets(IdxType idx) = 0;

        virtual char *get_vector(IdxType idx) = 0;

        virtual std::vector<LabelType> &get_label_set(IdxType idx) = 0;

        virtual inline void prefetch_vec_by_id(IdxType idx) const = 0;

        // obtain a point cloest to the center
        virtual IdxType choose_medoid(uint32_t num_threads, std::shared_ptr<DistanceHandler> distance_handler) = 0;

        // clean
        virtual void clean() = 0;
    };

    // storage class
    template<typename T>
    class Storage : public IStorage {

    public:
        Storage(DataType data_type, bool verbose) {
            this->data_type = data_type;
            this->verbose = verbose;
        }

        Storage(std::shared_ptr<IStorage> storage, IdxType start, IdxType end) {
            data_type = storage->get_data_type();
            num_points = end - start;
            dim = storage->get_dim();
            vecs = reinterpret_cast<T *>(storage->get_vector(start));
            label_sets = storage->get_offseted_label_sets(start);
            prefetch_byte_num = dim * sizeof(T);
            verbose = false;
        }

        ~Storage() = default;

        // I/O
        void load_from_file(const std::string &bin_file, const std::string &label_file, IdxType max_num_points) {
            if (verbose)
                std::cout << "Loading data from " << bin_file << " and " << label_file << " ..." << std::endl;
            auto start_time = std::chrono::high_resolution_clock::now();

            // open the binary file
            std::ifstream file(bin_file, std::ios::binary | std::ios::ate);
            if (!file.is_open())
                throw std::runtime_error("Failed to open file: " + bin_file);

            // read vector data
            file.seekg(0, std::ios::beg);
            file.read((char *) &num_points, sizeof(IdxType));
            file.read((char *) &dim, sizeof(IdxType));
            num_points = std::min(num_points, max_num_points);
            vecs = static_cast<T *>(std::aligned_alloc(32, num_points * dim * sizeof(T)));
            file.read((char *) vecs, num_points * dim * sizeof(T));
            file.close();

            // for prefetch
            prefetch_byte_num = dim * sizeof(T);

            // read label data if exists
            std::map<LabelType, IdxType> label_cnts;
            label_sets = new std::vector<LabelType>[num_points];
            file.open(label_file);
            if (file.is_open()) {
                std::string line, label;
                for (auto i = 0; i < num_points && i < max_num_points; ++i) {
                    std::getline(file, line);
                    std::stringstream ss(line);
                    while (std::getline(ss, label, ',')) {
                        label_sets[i].emplace_back(std::stoi(label));
                        if (label_cnts.find(std::stoi(label)) == label_cnts.end())
                            label_cnts[std::stoi(label)] = 1;
                        else
                            label_cnts[std::stoi(label)]++;
                    }
                    std::sort(label_sets[i].begin(), label_sets[i].end());
                    label_sets[i].shrink_to_fit();
                }
                file.close();

                // unfiltered ANNS when label file not found
            } else {
                std::cout << "- Warning: label file not found, set all labels to 1" << std::endl;
                for (auto i = 0; i < num_points && i < max_num_points; ++i)
                    label_sets[i] = {1};
                label_cnts[1] = num_points;
            }

            // statistics
            if (verbose) {
                std::cout << "- Number of points: " << num_points << std::endl;
                std::cout << "- Dimension: " << dim << std::endl;
                std::cout << "- Number of labels: " << label_cnts.size() << std::endl;
                std::cout << "- Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::high_resolution_clock::now() - start_time).count() << " ms" << std::endl;
            }
        }

        void write_to_file(const std::string &bin_file, const std::string &label_file) {
            // write vector data
            std::ofstream file(bin_file, std::ios::binary);
            file.write((char *) &num_points, sizeof(IdxType));
            file.write((char *) &dim, sizeof(IdxType));
            file.write((char *) vecs, num_points * dim * sizeof(T));
            file.close();

            // write label data
            file.open(label_file);
            for (auto i = 0; i < num_points; ++i) {
                file << label_sets[i][0];
                for (auto j = 1; j < label_sets[i].size(); ++j)
                    file << "," << label_sets[i][j];
                file << std::endl;
            }
            file.close();
        }


        // reorder the vector data
        void reorder_data(const std::vector<IdxType> &new_to_old_ids) {

            // move the vectors and labels
            auto new_vecs = static_cast<T *>(std::aligned_alloc(32, num_points * dim * sizeof(T)));
            auto new_label_sets = new std::vector<LabelType>[num_points];
            for (auto i = 0; i < num_points; ++i) {
                std::memcpy(new_vecs + i * dim, vecs + new_to_old_ids[i] * dim, dim * sizeof(T));
                new_label_sets[i] = label_sets[new_to_old_ids[i]];
            }

            // clean up
            delete[] vecs;
            delete[] label_sets;
            vecs = new_vecs;
            label_sets = new_label_sets;
        }

        // get statistics
        DataType get_data_type() const { return data_type; };

        IdxType get_num_points() const { return num_points; };

        IdxType get_dim() const { return dim; };

        // get data
        std::vector<LabelType> *get_offseted_label_sets(IdxType idx) { return label_sets + idx; }

        char *get_vector(IdxType idx) { return reinterpret_cast<char *>(vecs + idx * dim); }

        std::vector<LabelType> &get_label_set(IdxType idx) { return label_sets[idx]; }

        inline void prefetch_vec_by_id(IdxType idx) const {
            for (size_t d = 0; d < prefetch_byte_num; d += 64)
                _mm_prefetch((const char *) (vecs + idx * dim) + d, _MM_HINT_T0);
        }

        // obtain a point cloest to the center
        IdxType choose_medoid(uint32_t num_threads, std::shared_ptr<DistanceHandler> distance_handler) {

            // compute center
            T *center = new T[dim]();
            for (auto id = 0; id < num_points; ++id)
                for (auto d = 0; d < dim; ++d)
                    center[d] += *(vecs + id * dim + d);
            for (auto d = 0; d < dim; ++d)
                center[d] /= num_points;

            // obtain the closet point to the center
            std::vector<float> dists(num_points);
            omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 2048)
            for (auto id = 0; id < num_points; ++id)
                dists[id] = distance_handler->compute((const char *) center, get_vector(id), dim);
            IdxType medoid = std::min_element(dists.begin(), dists.end()) - dists.begin();

            // clean up
            delete[] center;
            return medoid;
        }

        // clean
        void clean() {
            if (vecs)
                delete[] vecs;
            if (label_sets)
                delete[] label_sets;
        }

    private:
        DataType data_type;
        IdxType num_points, dim;
        T *vecs = nullptr;
        size_t prefetch_byte_num;
        std::vector<LabelType> *label_sets = nullptr;

        // for logs
        bool verbose;
    };

    // obtain corresponding storage class
    std::shared_ptr<IStorage> create_storage(const std::string &data_type, bool verbose = true) {
        if (data_type == "float")
            return std::make_shared<Storage<float>>(DataType::FLOAT, verbose);
        else if (data_type == "int8")
            return std::make_shared<Storage<int8_t>>(DataType::INT8, verbose);
        else if (data_type == "uint8")
            return std::make_shared<Storage<uint8_t>>(DataType::UINT8, verbose);
        else {
            std::cerr << "Error: invalid data type " << data_type << std::endl;
            exit(-1);
        }
    }

    std::shared_ptr<IStorage> create_storage(std::shared_ptr<IStorage> storage, IdxType start, IdxType end) {
        DataType data_type = storage->get_data_type();
        if (data_type == DataType::FLOAT)
            return std::make_shared<Storage<float>>(storage, start, end);
        else if (data_type == DataType::INT8)
            return std::make_shared<Storage<int8_t>>(storage, start, end);
        else if (data_type == DataType::UINT8)
            return std::make_shared<Storage<uint8_t>>(storage, start, end);
        else {
            std::cerr << "Error: invalid data type " << data_type << std::endl;
            exit(-1);
        }
    }
}


#endif // ANNS_STORAGE_H