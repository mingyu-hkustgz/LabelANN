//
// Created by mingyu on 23-8-21.
//
#pragma once

#include <chrono>
#include <queue>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <unordered_map>
#include <cassert>
#include <chrono>
#include <memory>
#include <cstring>
#include <random>
#include <sstream>
#include <stack>
#include <x86intrin.h>
#include <immintrin.h>
#include <malloc.h>
#include <set>
#include <map>
#include <cmath>
#include <queue>
#include "hnswlib/hnswlib.h"
#include "config.h"

#ifndef WIN32

#include<sys/resource.h>

#endif
#define RANG_BOUND 1024
#define EPS_GROUND 1e-4
#define ID_OffSET 32 // ID OFFSET COMPRESS ID and Label Bitmap by split 64bit ID into [Bitmap][ID]
struct Neighbor {
    unsigned id;
    float distance;
    bool flag;

    Neighbor() = default;

    Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f) {}

    inline bool operator<(const Neighbor &other) const {
        return distance < other.distance;
    }
};

bool endsWith(const std::string &str, const std::string &suffix) {
    if (str.length() < suffix.length()) {
        return false;
    }
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

void write_kv_file(const std::string &filename, const std::map<std::string, std::string> &kv_map) {
    std::ofstream out(filename);
    for (auto &kv: kv_map) {
        out << kv.first << "=" << kv.second << std::endl;
    }
    out.close();
}


std::map<std::string, std::string> parse_kv_file(const std::string &filename) {
    std::map<std::string, std::string> kv_map;
    std::ifstream in(filename);
    std::string line;
    while (std::getline(in, line)) {
        size_t pos = line.find("=");
        if (pos == std::string::npos)
            continue;
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        kv_map[key] = value;
    }
    in.close();
    return kv_map;
}

bool isFileExists_ifstream(const char *name) {
    std::ifstream f(name);
    return f.good();
}

void write_gt_file(const std::string &filename, const std::pair<ANNS::IdxType, float> *gt, uint32_t num_queries,
                   uint32_t K) {
    std::ofstream fout(filename, std::ios::binary);
    fout.write(reinterpret_cast<const char *>(gt), num_queries * K * sizeof(std::pair<ANNS::IdxType, float>));
    std::cout << "Ground truth written to " << filename << std::endl;
}

void write_query_selectivity_file(const std::string &filename, const std::vector<double> &query_selectivity) {
    std::cout << "Writing query selectivity to " << filename << std::endl;
    std::ofstream fout(filename);
    float avg_selectivity = 0;
    fout << std::fixed;
    for (const auto &val: query_selectivity) {
        fout << val << std::endl;
        avg_selectivity += val;
    }
    fout << "avg selectivity: " << avg_selectivity / query_selectivity.size() << std::endl;
    fout.close();
}


void load_gt_file(const std::string &filename, std::pair<ANNS::IdxType, float> *gt, uint32_t num_queries, uint32_t K) {
    std::ifstream fin(filename, std::ios::binary);
    fin.read(reinterpret_cast<char *>(gt), num_queries * K * sizeof(std::pair<ANNS::IdxType, float>));
    std::cout << "Ground truth loaded from " << filename << std::endl;
}


float calculate_recall(const std::pair<ANNS::IdxType, float> *gt, const std::pair<ANNS::IdxType, float> *results,
                       uint32_t num_queries, uint32_t K) {
    float total_correct = 0;
    for (uint32_t i = 0; i < num_queries; i++) {

        // prepare ground truth set, offset records the last valid gt index
        std::set<ANNS::IdxType> gt_set;
        int32_t offset = -1;
        for (uint32_t j = 0; j < K; j++)
            if (gt[i * K + j].first != -1) {
                offset = j;
                gt_set.insert(gt[i * K + j].first);
            }

        // count the correct
        for (uint32_t j = 0; j < K; j++) {
            if (results[i * K + j].first == -1)
                break;
            if (offset >= 0 && results[i * K + j].second == gt[i * K + offset].second) {           // for ties
                total_correct++;
                offset--;
            } else {
                if (gt_set.find(results[i * K + j].first) != gt_set.end())
                    total_correct++;
            }
        }
    }
    return 100.0 * total_correct / (num_queries * K);
}

typedef std::priority_queue<std::pair<float, hnswlib::labeltype>> ResultQueue;


inline float sqr_dist(const float *d, const float *q, uint32_t L) {
    float PORTABLE_ALIGN32 TmpRes[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t num_blk16 = L >> 4;
    uint32_t l = L & 0b1111;

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);
    for (int i = 0; i < num_blk16; i++) {
        v1 = _mm256_loadu_ps(d);
        v2 = _mm256_loadu_ps(q);
        d += 8;
        q += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        v1 = _mm256_loadu_ps(d);
        v2 = _mm256_loadu_ps(q);
        d += 8;
        q += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }
    for (int i = 0; i < l / 8; i++) {
        v1 = _mm256_loadu_ps(d);
        v2 = _mm256_loadu_ps(q);
        d += 8;
        q += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }
    _mm256_store_ps(TmpRes, sum);

    float ret = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    for (int i = 0; i < l % 8; i++) {
        float tmp = (*q) - (*d);
        ret += tmp * tmp;
        d++;
        q++;
    }
    return ret;
}


void load_bitmap(const char *filename, std::vector<uint64_t> &label_bitmap, unsigned points_num) {
    if (!isFileExists_ifstream(filename)) {
        std::cout << "Label File Not Exists" << std::endl;
        assert(0);
        return;
    }
    unsigned cumulate_points = 0;
    std::ifstream fin(filename);
    std::string line;
    label_bitmap.reserve(points_num);
    unsigned max_bits = 0;
    while (std::getline(fin, line)) {
        cumulate_points++;
        uint64_t bitmap = 0;
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(iss, token, ',')) {
            tokens.push_back(token);
        }
        for (const auto &t: tokens) {
            bitmap |= (1ull << (std::atoi(t.c_str()) - 1));
        }
        label_bitmap.push_back(bitmap);
        if (_mm_popcnt_u64(bitmap) > max_bits) max_bits = _mm_popcnt_u64(bitmap);
        if (cumulate_points == points_num) break;
    }
    std::cerr << "Max bits:" << max_bits << std::endl;
#ifdef ID_COMPACT
    if(max_bits>ID_OffSET) std::cerr<<"EROOR ID_COMPACT WORKS ONLY FOR LESS THAN ID_OffSET BITS"<<std::endl;
#endif
}

void load_float_data(char *filename, float *&data, unsigned &num,
                     unsigned &dim) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *) &dim, 4);
    std::cout << "data dimension: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t) ss;
    num = (unsigned) (fsize / (dim + 1) / 4);
    data = new float[num * dim];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char *) (data + i * dim), dim * 4);
    }
    in.close();
}

void load_int_data(char *filename, int *&data, unsigned &num,
                   unsigned &dim) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *) &dim, 4);
    std::cout << "data dimension: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t) ss;
    num = (unsigned) (fsize / (dim + 1) / 4);
    data = new int[num * dim];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char *) (data + i * dim), dim * 4);
    }
    in.close();
}

float naive_l2_dist_calc(const float *q, const float *p, const unsigned &dim) {
    float ans = 0.0;
    for (unsigned i = 0; i < dim; i++) {
        ans += (p[i] - q[i]) * (p[i] - q[i]);
    }
    return ans;
}

float naive_inner_product(const float *q, const float *p, const unsigned &dim) {
    float ans = 0.00;
    for (unsigned i = 0; i < dim; i++) {
        ans += p[i] * q[i];
    }
    return ans;
}

int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn) {
    // find the location to insert
    int left = 0, right = K - 1;
    if (addr[left].distance > nn.distance) {
        memmove((char *) &addr[left + 1], &addr[left], K * sizeof(Neighbor));
        addr[left] = nn;
        return left;
    }
    if (addr[right].distance < nn.distance) {
        addr[K] = nn;
        return K;
    }
    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (addr[mid].distance > nn.distance)right = mid;
        else left = mid;
    }
    //check equal ID

    while (left > 0) {
        if (addr[left].distance < nn.distance) break;
        if (addr[left].id == nn.id) return K + 1;
        left--;
    }
    if (addr[left].id == nn.id || addr[right].id == nn.id)return K + 1;
    memmove((char *) &addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
    addr[right] = nn;
    return right;
}

float getRatio(const std::pair<ANNS::IdxType, float> *gt, const std::pair<ANNS::IdxType, float> *results,
               uint32_t num_queries, uint32_t K) {
    long double ret = 0;
    int valid_k = 0;
    for (uint32_t i = 0; i < num_queries; i++) {
        for (int j = 0; j < K; j++) {
            if (gt[i * K + j].second > 1e-5) {
                ret += std::sqrt(results[i * K + j].second / gt[i * K + j].second);
                valid_k++;
            }
        }
    }
    if (valid_k == 0) return 1.0;
    return ret / valid_k;
}


#ifndef WIN32

void GetCurTime(rusage *curTime) {
    int ret = getrusage(RUSAGE_SELF, curTime);
    if (ret != 0) {
        fprintf(stderr, "The running time info couldn't be collected successfully.\n");
        //FreeData( 2);
        exit(0);
    }
}

/*
* GetTime is used to get the 'float' format time from the start and end rusage structure.
*
* @Param timeStart, timeEnd indicate the two time points.
* @Param userTime, sysTime get back the time information.
*
* @Return void.
*/
void GetTime(struct rusage *timeStart, struct rusage *timeEnd, float *userTime, float *sysTime) {
    (*userTime) = ((float) (timeEnd->ru_utime.tv_sec - timeStart->ru_utime.tv_sec)) +
                  ((float) (timeEnd->ru_utime.tv_usec - timeStart->ru_utime.tv_usec)) * 1e-6;
    (*sysTime) = ((float) (timeEnd->ru_stime.tv_sec - timeStart->ru_stime.tv_sec)) +
                 ((float) (timeEnd->ru_stime.tv_usec - timeStart->ru_stime.tv_usec)) * 1e-6;
}

#endif

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))

#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif


/**
* Returns the peak (maximum so far) resident set size (physical
* memory use) measured in bytes, or zero if the value cannot be
* determined on this OS.
*/
size_t getPeakRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
        return (size_t)0L;      /* Can't open? */
    if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo))
    {
        close(fd);
        return (size_t)0L;      /* Can't read? */
    }
    close(fd);
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t) (rusage.ru_maxrss * 1024L);
#endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L;          /* Unsupported. */
#endif
}


/**
* Returns the current resident set size (physical memory use) measured
* in bytes, or zero if the value cannot be determined on this OS.
*/
size_t getCurrentRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
        (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L;      /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t) 0L;      /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t) 0L;      /* Can't read? */
    }
    fclose(fp);
    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;          /* Unsupported. */
#endif
}


inline void print_error_and_terminate(std::stringstream &error_stream) {
    std::cerr << error_stream.str() << std::endl;
}

inline void report_misalignment_of_requested_size(size_t align) {
    std::stringstream stream;
    stream << "Requested memory size is not a multiple of " << align << ". Can not be allocated.";
    print_error_and_terminate(stream);
}

inline void report_memory_allocation_failure() {
    std::stringstream stream;
    stream << "Memory Allocation Failed.";
    print_error_and_terminate(stream);
}


inline void check_stop(std::string arnd) {
    int brnd;
    std::cout << arnd << std::endl;
    std::cin >> brnd;
}

inline void aligned_free(void *ptr) {
    // Gopal. Must have a check here if the pointer was actually allocated by
    // _alloc_aligned
    if (ptr == nullptr) {
        return;
    }
#ifndef _WINDOWS
    free(ptr);
#else
    ::_aligned_free(ptr);
#endif
}


