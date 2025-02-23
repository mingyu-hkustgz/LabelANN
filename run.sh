mkdir build
cd build
cmake ..
make -j 48

cd ..
./build/test/test_hnsw_build -d sift -s ./DATA/sift/

./build/test/test_hnsw_search -d sift -s ./DATA/sift/
