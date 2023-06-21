. ./tools/activate_python.sh

db_dir=/DB/librispeech_finetuning/1h/1
data_dir=$PWD/datas/train_10m/
ext=flac
valid=0.0

python examples/wav2vec/wav2vec_manifest.py $db_dir --dest $data_dir --ext $ext --valid-percent $valid

split=train
python examples/wav2vec/libri_labels.py $data_dir/train.tsv --output-dir $data_dir --output-name $split

cp -r $PWD/datas/train_clean_100/test* $data_dir
cp -r $PWD/datas/train_clean_100/dev* $data_dir
cp -r $PWD/datas/train_clean_100/valid* $data_dir
cp -r $PWD/datas/train_clean_100/dict.ltr.txt $data_dir
