#! /bin/bash
# =========目录结构==========
# /home/ai/data/
# --------------VOCdevkit/VOC2007 or VOC2012
# ------------------------Annotations/      .xml .xml .xml ...
# ------------------------ImageSets/Main/   trainval.txt test.txt
# ------------------------JPEGImages/       .jpg .jpg .jpg ...
# ==========================
# 生成 /home/ai/ssd/caffe/data/VOC0712/trainval.txt 和 test.txt
root_dir=$HOME/data/VOCdevkit/
sub_dir=ImageSets/Main
# 该bash脚本路径 bash_dir = /home/ai/ssd/caffe/data/VOC0712
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 生成trainval.txt 和 test.txt
# .../VOC0712/
# -----------/trainval.txt test.txt
for dataset in trainval test
do
  dst_file=$bash_dir/$dataset.txt
  if [ -f $dst_file ]
  then
    rm -f $dst_file
  fi
  for name in VOC2007 VOC2012
  do
    # VOC2012数据集没有test
    if [[ $dataset == "test" && $name == "VOC2012" ]]
    then
      continue
    fi
    echo "Create list for $name $dataset..."
    # /hoom/ai/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt
    dataset_file=$root_dir/$name/$sub_dir/$dataset.txt

    # /home/ai/ssd/caffe/data/VOC0712/trainval_img.txt
    img_file=$bash_dir/$dataset"_img.txt"

    # /hoom/ai/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt 复制到
    # /home/ai/ssd/caffe/data/VOC0712/trainval_img.txt
    cp $dataset_file $img_file
    # 修改 trainval_img.txt中的图像路径
    # 在每一行开头添加 VOC2007/JPEGImages/
    sed -i "s/^/$name\/JPEGImages\//g" $img_file
    # 在每一行行尾添加.jpg
    sed -i "s/$/.jpg/g" $img_file
    # 得到trainval_img.txt文件，每一行格式为 VOC2007/JPEGImages/1.jpg

    label_file=$bash_dir/$dataset"_label.txt"
    cp $dataset_file $label_file
    sed -i "s/^/$name\/Annotations\//g" $label_file
    sed -i "s/$/.xml/g" $label_file
    # 得到trainval_label.txt文件，每一行格式为 VOC2007/Annotations/1.xml

    paste -d' ' $img_file $label_file >> $dst_file
    # 得到trainval.txt 和 test.txt
    # VOC2007/JPEGImages/1.jpg VOC2007/Annotations/1.xml
    # VOC2012/JPEGImages/1.jpg VOC2012/Annotations/1.xml

    # 删除临时文件
    rm -f $label_file
    rm -f $img_file
  done

  # Generate image name and size infomation.
  # 生成test数据集的name和size信息文件，测试图像路径文件 /home/ai/ssd/caffe/data/VOC0712/test.txt
  # /home/ai/ssd/caffe/data/VOC0712/VOC2007_name_size.txt
  if [ $dataset == "test" ]
  then
    # /home/ai/ssd/caffe/get_image_size /home/ai/data/VOCdevkit/ /home/ai/ssd/caffe/data/VOC0712/test.txt /home/ai/ssd/caffe/data/VOC0712/VOC2007_name_size.txt
    $bash_dir/../../build/tools/get_image_size $root_dir $dst_file $bash_dir/$dataset"_name_size.txt"
  fi

  # Shuffle trainval file.
  # 对整个trainval.txt洗牌，打乱图像顺序
  if [ $dataset == "trainval" ]
  then
    # /home/ai/ssd/caffe/data/VOC0712/trainval.txt.random
    rand_file=$dst_file.random
    cat $dst_file | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > $rand_file
    mv $rand_file $dst_file
    echo "shuffled"
  fi
done
