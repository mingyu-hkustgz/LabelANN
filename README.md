# Elastic Label Index: Efficient Approximate Nearest Neighbor Search with Label Selected

This is the official implementation of the paper Elastic Label Index: Efficient Approximate Nearest Neighbor Search with Label Selected.
ELI selectively indexing only a subset of query-label sets while still ensuring efficient processing for all
queries. We prove the problem is NP-complete and propose efficient greedy algo-
rithms for its efficiency- and space-constrained variants. Extensive
experiments on real-world datasets show that our method achieves
10×–800× speedups over state-of-the-art techniques.

## Requirement
* C++
  * Cmake
  * OpenMP
  * Boost
* Hardware Support
  * AVX512
  * AVX
  * SSE

## Repository Structure
```
.
├── CMakeLists.txt
├── README.md
├── figure
│   ├── Example.png
│   ├── dynamic.png
│   └── unseen.png
├── include                     # ELI source codes
├── mkdir.sh
├── run.sh
├── script                      # test script
├── set.sh
├── technical_report.pdf
├── test                        # C++ codes for the test script
└── tools                       # util functions to generate labels and compute groundtruth
```

## Reproduction
1. set the `store_path` and `datasets`, `labelsets` variables in set.sh. 
    `store_path` is the path to store the dataset, `datasets` is the list of dataset names, `labelsets` is the list of label set names. 
    Predefined labelsets are `multi_normial/zipf/uniform/poisson/one_per_point` (generate by `pre_process.sh`), but you can also provide your own labelsets.
2. run `bash run.sh`. Refer to docs in the scripts for more options.

## Data Path
1. All dataset store in fvecs format or bin format.
2. The dataset path needs to be arranged as
```
{store_path}/{dataset_name}/
   -- {dataset_name}_base.[bin]&[fvecs]
   -- {dataset_name}_query.[bin]&[fvecs]
```
   
Example:
```
./DATA
├── sift
│   ├── sift_base.fvecs
│   └── sift_query.fvecs
```

## Example of Index Selection
![Greedy Algorithm Running Example](./figure/Example.png "Greedy Result")

## Example of Dynamic Index
![Index Adjust Example](./figure/dynamic.png "Dynamic Index")

![Handle Unseen Label Example](./figure/unseen.png "Unseen Label")

