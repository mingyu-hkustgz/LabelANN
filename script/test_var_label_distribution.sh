source set.sh

for data in "${datasets[@]}"; do
  if [ $data == "sift" ]; then
      N=1000000
      QN=10000
  elif [ $data == "OpenAI-1536" ]; then
      N=999000
      QN=1000
  elif [ $data == "OpenAI-3072" ]; then
      N=999000
      QN=1000
  elif [ $data == "yt1m" ]; then
      N=999000
      QN=1000
  elif [ $data == "msmarc-small" ]; then
      N=1000000
      QN=1000
  elif [ $data == "paper" ]; then
      N=2029997
      QN=10000
  elif [ $data == "deep100M" ]; then
      N=100000000
      QN=1000
  fi

  L=12

  for label in "${labelsets[@]}"; do

  ./build/tools/generate_base_labels \
      --num_labels $L --num_points $N --distribution_type ${label} \
      --output_file ./DATA/${data}/${data}_base_${L}_labels_${label}.txt

  ./build/tools/generate_query_labels \
      --num_points $QN --distribution_type ${label} --K 10 --scenario containment \
      --input_file ./DATA/${data}/${data}_base_${L}_labels_${label}.txt \
      --output_file ./DATA/${data}/${data}_query_${L}_labels_${label}_containment.txt

  ./build/tools/compute_groundtruth \
      --data_type float --dist_fn L2 --scenario containment --K 10 --num_threads 144 \
      --base_bin_file ./DATA/${data}/${data}_base.fvecs \
      --base_label_file ./DATA/${data}/${data}_base_${L}_labels_${label}.txt \
      --query_bin_file ./DATA/${data}/${data}_query.fvecs \
      --query_label_file ./DATA/${data}/${data}_query_${L}_labels_${label}_containment.txt \
      --gt_file ./DATA/${data}/${data}_gt_${L}_labels_${label}_containment.bin


  log_file="./results/time-log/${data}/HNSW-Elastic-Auto-Index-time.log"
  start_time=$(date +%s)
  ./build/test/test_elastic_hnsw_build_compact -d ${data} -s ./DATA/${data}/ -l ${L}_labels_${label} -e  2.0
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "Elastic[2.0] L=${L} HNSW Index time(${label}): ${duration}(s)" | tee -a ${log_file}

  ./build/test/test_elastic_hnsw_search_compact -d ${data} -s ./DATA/${data}/ -l ${L}_labels_${label} -e 2.0


  log_file="./results/time-log/${data}/HNSW-Elastic-Index-time.log"
  start_time=$(date +%s)
  ./build/test/test_elastic_hnsw_build_compact -d ${data} -s ./DATA/${data}/ -l ${L}_labels_${label} -e 0.2
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "Elastic[0.2] L=${L} HNSW Index time(${label}): ${duration}(s)" | tee -a ${log_file}

  ./build/test/test_elastic_hnsw_search_compact -d ${data} -s ./DATA/${data}/ -l ${L}_labels_${label} -e 0.2


  done
done