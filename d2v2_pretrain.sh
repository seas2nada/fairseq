#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

data_dir=$PWD/datas/train_960/
ngpu=4
update_freq=$((8/$ngpu))
config_dir=$PWD/examples/data2vec/config/v2
config_name=base_audio_only_task

. tools/activate_python.sh

# make sure to set model save_dir path in wav2vec2_large_librivox.yaml
# checkpoint:
#   save_dir: /path/to/model/save/dir

export PYTHONPATH=$PWD
export PYTHONPATH=$PYTHONPATH:$PWD/examples/data2vec
FAIRDIR=$PWD
python $FAIRDIR/fairseq_cli/hydra_train.py -m \
    task.data=$data_dir \
    common.user_dir=$PWD/examples/data2vec \
    distributed_training.distributed_world_size=$ngpu +optimization.update_freq='['$update_freq']' \
    --config-dir $config_dir \
    --config-name $config_name
