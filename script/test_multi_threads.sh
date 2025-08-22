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
  elif [ $data == "msmarc-small" ]; then
      N=1000000
  elif [ $data == "paper" ]; then
      N=2029997
  elif [ $data == "deep100M" ]; then
      N=100000000
  elif [ $data == "arxiv-for-fanns-large" ]; then
      N=2735264
  fi
  for L in 12;do
      log_file="./results/time-log/${data}/HNSW-Elastic-Index-time.log"
      start_time=$(date +%s)
      ./build/test/test_elastic_hnsw_build_compact -d ${data} -s ./DATA/${data}/ -l ${L}_labels_zipf -e 0.2
      end_time=$(date +%s)
      duration=$((end_time - start_time))
      echo "Elastic[0.2] L=${L} HNSW Index time: ${duration}(s)" | tee -a ${log_file}

      log_file="./results/time-log/${data}/HNSW-Elastic-Index-time.log"
      start_time=$(date +%s)
      ./build/test/test_elastic_hnsw_build_compact -d ${data} -s ./DATA/${data}/ -l ${L}_labels_zipf -e 2.0
      end_time=$(date +%s)
      duration=$((end_time - start_time))
      echo "Elastic[2.0] L=${L} HNSW Index time: ${duration}(s)" | tee -a ${log_file}


      for T in {1,4,8,16,32};do
      ./build/test/test_elastic_hnsw_search_compact -d ${data} -s ./DATA/${data}/ -l ${L}_labels_zipf -e 0.2 -t $T

      ./build/test/test_elastic_hnsw_search_compact -d ${data} -s ./DATA/${data}/ -l ${L}_labels_zipf -e 2.0 -t $T
    done
  done
done