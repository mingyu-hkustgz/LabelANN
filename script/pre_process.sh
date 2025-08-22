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
  elif [ $data == "laion" ]; then
      N=1182243
      QN=1000
  elif [ $data == "TripClick" ]; then
      N=1020825
      QN=1000
  elif [ $data == "paper" ]; then
      N=2029997
      QN=10000
  elif [ $data == "deep100M" ]; then
      N=100000000
      QN=1000
  fi
  for L in {8,12,24,32};do

  ./build/tools/generate_base_labels \
      --num_labels $L --num_points $N --distribution_type zipf \
      --output_file ./DATA/${data}/${data}_base_${L}_labels_zipf.txt

  ./build/tools/generate_query_labels \
      --num_points $QN --distribution_type zipf --K 10 --scenario containment \
      --input_file ./DATA/${data}/${data}_base_${L}_labels_zipf.txt \
      --output_file ./DATA/${data}/${data}_query_${L}_labels_zipf_containment.txt

  ./build/tools/compute_groundtruth \
      --data_type float --dist_fn L2 --scenario containment --K 10 --num_threads 144 \
      --base_bin_file ./DATA/${data}/${data}_base.fvecs \
      --base_label_file ./DATA/${data}/${data}_base_${L}_labels_zipf.txt \
      --query_bin_file ./DATA/${data}/${data}_query.fvecs \
      --query_label_file ./DATA/${data}/${data}_query_${L}_labels_zipf_containment.txt \
      --gt_file ./DATA/${data}/${data}_gt_${L}_labels_zipf_containment.bin

  done
done