## Dataset

The overall directory structure should be:
```
│PointMamba/
├──cfgs/
├──data/
│   ├──ModelNet/
│   ├──ScanObjectNN/
│   ├──ShapeNet55-34/
│   ├──shapenetcore_partanno_segmentation_benchmark_v0_normal/
├──datasets/
├──.......
```

### ModelNet40 Dataset: 

```
│ModelNet/
├──modelnet40_normal_resampled/
│  ├── modelnet40_shape_names.txt
│  ├── modelnet40_train.txt
│  ├── modelnet40_test.txt
│  ├── modelnet40_train_8192pts_fps.dat
│  ├── modelnet40_test_8192pts_fps.dat
```

Download: You can download the processed data from [Point-BERT repo](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md), or download from the [official website](https://modelnet.cs.princeton.edu/#) and process it by yourself.

### ScanObjectNN Dataset:

```
│ScanObjectNN/
├──main_split/
│  ├── training_objectdataset_augmentedrot_scale75.h5
│  ├── test_objectdataset_augmentedrot_scale75.h5
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
├──main_split_nobg/
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
```
Download: Please download the data from the [official website](https://hkust-vgd.github.io/scanobjectnn/).

### ShapeNet55/34 Dataset:

```
│ShapeNet55-34/
├──shapenet_pc/
│  ├── 02691156-1a04e3eab45ca15dd86060f189eb133.npy
│  ├── 02691156-1a6ad7a24bb89733f412783097373bdc.npy
│  ├── .......
├──ShapeNet-55/
│  ├── train.txt
│  └── test.txt
```

Download: Please download the data from [Point-BERT repo](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md).
