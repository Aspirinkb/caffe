cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=$cur_dir/../..
# /home/ai/ssd/caffe/data/Inspur/

cd $root_dir

redo=1
data_root_dir="/home/hh/v2i/data/image/2_persecond"
dataset_name="Inspur"
mapfile="/home/ai/ssd/caffe/data/Inspur/labelmap_inspur.prototxt"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0
lmdb_dir="/home/ai/data/Inspur"

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in test trainval
do
  python3 $root_dir/scripts/create_annoset_inspur.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $root_dir/data/$dataset_name/$subset.txt $lmdb_dir/$dataset_name"_"$subset"_"$db examples/$dataset_name
  # python /home/ai/ssd/caffe/scripts/create_annoset.py
  # --anno_type=detection
  # --label-map-file=/home/ai/ssd/caffe/data/Inspur/labelmap_inspur.prototxt
  # --min-dim --max-dim --resize-width --resize-height
  # --check-label
  # --encode-type=jpg
  # --encoded
  # --redo
  # /home/hh/v2i/data/image/2_persecond/ home/ai/ssd/caffe/data/Inspur/trainval.txt home/data/Inspur/lmdb/Inspur_trainval_lmdb examples/Inspur
  # /home/hh/v2i/data/image/2_persecond/ home/ai/ssd/caffe/data/Inspur/test.txt home/data/Inspur/lmdb/Inspur_test_lmdb examples/Inspur
done
