#!/bin/bash

#e.g. bash ./src/eval.sh yago_ko_0.4 0 1 -1 yago 7

time_start=$(date +"%Y%m%d_%H%M%S")

green=`tput setaf 2`
red=`tput setaf 1`
yellow=`tput setaf 3`
reset=`tput sgr0`

# input variables
model=$1 # which model to run: HEER, DistMult, ComplEx, TransE, node2vec, DeepWalk
entity_pred=${2:-true}
relation_pred=${3:-true}

# find relative root directory
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
script_dir="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
root_dir="$( dirname $script_dir )"/network-embedding-evaluation

echo ${yellow}===$model Testing===${reset}
ent_output_file="$root_dir"/results/"$model"_entity_prediction_metrics.txt
rel_output_file="$root_dir"/results/"$model"_relation_prediction_metrics.txt

if [[ "${entity_pred}" =~ "true" ]]; then
  echo "Computing entity predictions for all test cases..."
  python2 "$root_dir"/pred_head_tail.py --model=$model --output-file=$ent_output_file
fi
if [[ "${relation_pred}" =~ "true" ]]; then
  echo "Computing relation predictions for all test cases..."
  python2 "$root_dir"/pred_relation.py --model=$model --output-file=$rel_output_file
fi
echo "Done."
