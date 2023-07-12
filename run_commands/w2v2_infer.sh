#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

for subset in "dev_clean" "dev_other" "test_clean" "test_other"; do
    data_dir=$PWD/datas/$subset/
    finetuned_model=$PWD/models/w2v2_base_MPI_10m_1/0/checkpoints/checkpoint_l1_unst_pruned10.pt
    # finetuned_model=$PWD/models/wav2vec_small_10m_converted.pt
    inference_result=$PWD/inference_result/
    wordscore=-1
    lmweight=2
    silscore=0
    num_gpus=1

    . ./tools/activate_python.sh

    export PYTHONPATH=$PWD
    export PYTHONPATH=$PYTHONPATH:$PWD/examples/wav2vec
    FAIRDIR=$PWD

    python $FAIRDIR/examples/speech_recognition/new/infer.py --config-dir $FAIRDIR/examples/speech_recognition/new/conf \
    --config-name infer task=audio_finetuning task.data=$data_dir common.user_dir=$FAIRDIR/examples/wav2vec \
    task.labels=ltr decoding.type=viterbi \
    decoding.wordscore=${wordscore} decoding.silweight=${silscore} \
    decoding.unique_wer_file=True \
    dataset.gen_subset=$subset dataset.max_tokens=2500000 \
    common_eval.path=$finetuned_model decoding.beam=1500 distributed_training.distributed_world_size=${num_gpus}
done

# for subset in "dev_clean" "dev_other" "test_clean" "test_other"; do
#     data_dir=$PWD/datas/$subset/
#     finetuned_model=$PWD/models/w2v2_base_10m/0/checkpoints/checkpoint_best.pt
#     inference_result=$PWD/inference_result/
#     lexicon=$PWD/lm_downloads/librispeech_lexicon.lst
#     lm_model=$PWD/lm_downloads/4-gram.bin
#     wordscore=-1
#     lmweight=2
#     silscore=0
#     num_gpus=1

#     . ./tools/activate_python.sh

#     export PYTHONPATH=$PWD
#     export PYTHONPATH=$PYTHONPATH:$PWD/examples/data2vec
#     FAIRDIR=$PWD

#     python $FAIRDIR/examples/speech_recognition/new/infer.py --config-dir $FAIRDIR/examples/speech_recognition/new/conf \
#     --config-name infer task=audio_finetuning task.data=$data_dir common.user_dir=$FAIRDIR/examples/data2vec \
#     task.labels=ltr decoding.type=kenlm \
#     decoding.lmweight=${lmweight} decoding.wordscore=${wordscore} decoding.silweight=${silscore} \
#     decoding.lexicon=$lexicon \
#     decoding.lmpath=$lm_model decoding.unique_wer_file=True \
#     dataset.gen_subset=$subset dataset.max_tokens=2500000 \
#     common_eval.path=$finetuned_model decoding.beam=1500 distributed_training.distributed_world_size=${num_gpus}
# done
