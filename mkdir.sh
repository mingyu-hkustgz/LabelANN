source set.sh
mkdir ./DATA
mkdir ./results
mkdir ./results/time-log
mkdir ./results/space-log
mkdir ./results@1
mkdir ./results@10

rm -rf build
mkdir build
cd build

cmake ..
make clean
make -j 40

cd ..

mkdir ./figure

for dataset in "${datasets[@]}";
do
  echo $dataset
  mkdir ./DATA/${dataset}
  mkdir ./results@1/${dataset}
  mkdir ./results@10/${dataset}
  mkdir ./results/time-log/${dataset}
  mkdir ./results/space-log/${dataset}
  mkdir ./figure/${dataset}
done