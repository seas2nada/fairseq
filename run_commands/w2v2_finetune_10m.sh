#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

data_dir=$PWD/datas/train_10m_1/
# pretrained_model=$PWD/downloads/wav2vec_small.pt
pretrained_model=$PWD/models/w2v2_base/checkpoint_l1_unst_pruned10.pt
ngpu=1
config_dir=$PWD/examples/wav2vec/config/finetuning
config_name=base_10m

#. ./path.sh || exit 1;
#. utils/parse_options.sh || exit 1;
. ./tools/activate_python.sh

# make sure to set model save_dir path in vox_100h.yaml
# checkpoint:
#   save_dir: /path/to/model/save/dir
FAIRDIR=$PWD
export PYTHONPATH=$PWD
# export PYTHONPATH=$PYTHONPATH:$PWD/examples/wav2vec
python $FAIRDIR/fairseq_cli/hydra_train.py -m --config-dir $FAIRDIR/examples/wav2vec/config/finetuning --config-name $config_name \
	task.data=$data_dir model.w2v_path=$pretrained_model distributed_training.distributed_world_size=$ngpu optimization.update_freq="[8]"
