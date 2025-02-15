//
// Created by mingyu on 25-2-15.
//
#include <iostream>
#include <fstream>
#include <cstdio>
#include <getopt.h>
#include "Index-L.h"
using namespace std;


int main(int argc, char* argv[])
{
    const struct option longopts[] = {
        // General Parameter
        {"help", no_argument, 0, 'h'},

        // Indexing Path
        {"dataset", required_argument, 0, 'd'},
        {"source", required_argument, 0, 's'},
    };

    int ind;
    int iarg = 0;
    opterr = 1; //getopt error message (off: 0)

    char dataset[256] = "";
    char source[256] = "";
    char data_path[256] = "";
    char query_path[256] = "";
    char ground_path[256] = "";
    while (iarg != -1)
    {
        iarg = getopt_long(argc, argv, "d:s:", longopts, &ind);
        switch (iarg)
        {
        case 'd':
            if (optarg)
            {
                strcpy(dataset, optarg);
            }
            break;
        case 's':
            if (optarg)
            {
                strcpy(source, optarg);
            }
            break;
        }
    }
    sprintf(query_path, "%s%s_query.bin", source, dataset);
    sprintf(data_path, "%s%s_base.bin", source, dataset);
    sprintf(ground_path, "%s%s_groundtruth.ivecs", source, dataset);
    Matrix<float> X(data_path);
    Matrix<float> Q(query_path);
    Matrix<int> G(ground_path);
    std::vector<std::vector<int>> gt;
    gt.resize(Q.n);
    for (int i = 0; i < Q.n; i++){
        gt[i].resize(100);
        for (int j = 0; j < 100; j++){
            gt[i][j] = G.data[i * 100 + j];
        }
        std::cerr<<gt[i][0]<<" ";
    }

    IndexLabel hnswl(X.n, X.d);
    hnswl.build_index(X);
    std::vector efSearch{1, 2, 4, 8, 16, 32, 50, 64, 128, 150, 256, 300};
    std::ofstream fout("./results/sift/sift.log");
    for (auto ef : efSearch)
    {
        double segment_recall = 0.0, segment = 0.0, all_index_search_time = 0.0;
        for (int i = 0; i < Q.n; i++)
        {
            ResultQueue ans1;

            auto s = chrono::high_resolution_clock::now();
            ans1 = hnswl.naive_search(Q.data + i * Q.d, 10, ef);
            auto e = chrono::high_resolution_clock::now();
            chrono::duration<double> diff = e - s;
            double time_slap1 = diff.count();
            while (!ans1.empty())
            {
                auto v = ans1.top();
                if (std::find(gt[i].begin(), gt[i].end(), v.second) != gt[i].end())
                    segment += 1.0;
                ans1.pop();
            }
            segment /= 10;
            segment_recall += segment;
            all_index_search_time += time_slap1;
        }
        segment_recall /= (double)Q.n;
        double Qps = (double)Q.n / all_index_search_time;
        fout << segment_recall * 100 << " " << Qps << std::endl;
    }

    return 0;
}
