source set.sh


for data in "${datasets[@]}"; do
  if [ $data == "sift" ]; then
      N=1000000
  elif [ $data == "gist" ]; then
      N=1000000
  elif [ $data == "OpenAI-1536" ]; then
      N=9990000
  elif [ $data == "OpenAI-3072" ]; then
      N=9990000
  elif [ $data == "yt1m" ]; then
      N=999000
  elif [ $data == "msmarc" ]; then
      N=1000000
  elif [ $data == "paper" ]; then
      N=2029997
  elif [ $data == "deep100M" ]; then
      N=100000000
  fi
  for L in {8,12,24,32};do

  log_file="./results/time-log/${data}/HNSW-Elastic-Index-time.log"
  start_time=$(date +%s)
  ./build/test/test_elastic_hnsw_build_compact -d ${data} -s ./DATA/${data}/ -l ${L}_labels_zipf -e 0.2
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "Elastic[0.2] L=${L} HNSW Index time: ${duration}(s)" | tee -a ${log_file}

  ./build/test/test_elastic_hnsw_search_compact -d ${data} -s ./DATA/${data}/ -l ${L}_labels_zipf -e 0.2

  log_file="./results/time-log/${data}/HNSW-Elastic-Index-time.log"
  start_time=$(date +%s)
  ./build/test/test_elastic_hnsw_build_compact -d ${data} -s ./DATA/${data}/ -l ${L}_labels_zipf -e 0.5
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "Elastic[0.5] L=${L} HNSW Index time: ${duration}(s)" | tee -a ${log_file}

  ./build/test/test_elastic_hnsw_search_compact -d ${data} -s ./DATA/${data}/ -l ${L}_labels_zipf -e 0.5

  log_file="./results/time-log/${data}/HNSW-Optimal-Index-time.log"
  start_time=$(date +%s)
  ./build/test/test_opt_hnsw_build -d ${data} -s ./DATA/${data}/ -l ${L}_labels_zipf
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "Optimal L=${L} HNSW Index time: ${duration}(s)" | tee -a ${log_file}

  ./build/test/test_opt_hnsw_search -d ${data} -s ./DATA/${data}/ -l ${L}_labels_zipf

  done
done