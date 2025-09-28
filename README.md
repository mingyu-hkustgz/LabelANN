# Elastic Label Index: Efficient Approximate Nearest Neighbor Search with Label Selected

## Requirement
* C++
  * Cmake
  * OpenMP
  * Boost
* Hardware
  * AVX512
  * AVX
  * SSE

## Reproduction
1. set the store_path and dataset in set.sh
2. run bash run.sh

## Data Path
1. All dataset store in fvecs format or bin format.
2. The dataset path needs to be arranged as
   --{store_path}/{dataset_name}/
   --{dataset_name}_base.[bin]&[fvecs]
   --{dataset_name}_query.[bin]&[fvecs]
3. Example
 * ./DATA/sift/sift_base.fvecs
 * ./DATA/sift/sfit_query.fvecs

## Example of Index Select
![Greedy Algorithm Running Example](./figure/Runing-Example.pdf "Greedy Result")

## Example of Dynamic
![Index Adjust Example](./figure/dynamic-adjust.pdf "Dynamic Index")

![Handle Unseen Label Example](./figure/dynamic-unseen.pdf "Unseen Label")

