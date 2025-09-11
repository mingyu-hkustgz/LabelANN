source set.sh

for data in "${datasets[@]}"; do
  if [ $data == "sift" ]; then
      N=1000000
      QN=10000
      L=12
  elif [ $data == "OpenAI-1536" ]; then
      N=999000
      QN=1000
      L=12
  elif [ $data == "OpenAI-3072" ]; then
      N=999000
      QN=1000
      L=12
  elif [ $data == "yt1m" ]; then
      N=999000
      QN=1000
      L=12
  elif [ $data == "msmarc-small" ]; then
      N=1000000
      QN=1000
      L=12
  elif [ $data == "paper" ]; then
      N=2029997
      QN=10000
      L=12
  elif [ $data == "laion" ]; then
      N=1182243
      QN=1000
      L=12
  elif [ $data == "deep100M" ]; then
      N=100000000
      QN=1000
      L=12
  elif [ $data == "TripClick" ]; then
      N=1020825
      QN=1000
      L=12
  elif [ $data == "arxiv-for-fanns-large" ]; then
      N=2735264
      QN=1000
      L=12
  elif [ $data == "arxiv-for-fanns-medium" ]; then
      N=100000
      QN=1000
      L=12
  fi

  for label in "${labelsets[@]}"; do

#  ./build/tools/generate_base_labels \
#      --num_labels $L --num_points $N --distribution_type ${label} \
#      --output_file ./DATA/${data}/${data}_base_${L}_labels_${label}.txt
#
#  ./build/tools/generate_query_labels \
#      --num_points $QN --distribution_type ${label} --K 10 --scenario overlap \
#      --input_file ./DATA/${data}/${data}_base_${L}_labels_${label}.txt \
#      --output_file ./DATA/${data}/${data}_query_${L}_labels_${label}_overlap.txt
#
    ./build/test/test_elastic_hnsw_build -d ${data} -s ./DATA/${data}/ -l ${L}_labels_${label} -e 0.2

#    ./build/test/test_elastic_hnsw_build -d ${data} -s ./DATA/${data}/ -l ${L}_labels_${label} -e 2.0


    ./build/tools/local_test_overlap_groundtruth -d ${data} -s ./DATA/${data}/ -l ${L}_labels_${label} -k 10

#    ./build/test/test_elastic_overlap_search -d ${data} -s ./DATA/${data}/ -l ${L}_labels_${label} -e 2.0

    ./build/test/test_elastic_overlap_search -d ${data} -s ./DATA/${data}/ -l ${L}_labels_${label} -e 0.2

  done
done