# DL-Project

### Requirements
  - python3
  - pytorch==0.4.1
  - torchvision>=0.2.0
  - cython
  - matplotlib
  - numpy
  - scipy
  - opencv
  - pyyaml==3.12
  - packaging
  - pandas
  - pycocotools
  - tensorboardX
- Must use CUDA 9.0.

### Compilation

Compile the CUDA code:

```
cd lib  # please change to this directory
sh make.sh
```
### Data Preparation

Please add `data` in the `fsod` directory and the structure is :

  ```
  YOUR_PATH
      └── fsod
            ├── other files
            └── data
                  └──── fsod
                          ├── annotations
                          │       ├── fsod_train.json
                          │       └── fsod_test.json
                          └── images
                                ├── part_1
                                └── part_2
  ```  
  
### Training and evaluation

```
CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --save_dir fsod_save_dir --dataset fsod --cfg configs/fsod/voc_e2e_faster_rcnn_R-50-C4_1x_old_1.yaml --bs 1 --iter_size 2 --nw 4

CUDA_VISIBLE_DEVICES=0 python3 tools/test_net.py --dataset fsod --cfg configs/fsod/voc_e2e_faster_rcnn_R-50-C4_1x_old_1.yaml --load_ckpt Outputs/fsod_save_dir/ckpt/model_step59999.pth
```
  
