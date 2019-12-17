# Automated Pollen Detection with an Affordable Technology
This repository contains the code and hardware design files accompanying the paper [tbd](). The data can be found [here](https://zenodo.org/record/3572653) (download instructions provided below).

## Hardware designs
The [./hwdesign/](./hwdesign/) folder contains the design files for the automated pollen trap. For more information please refer to the paper.

## Code
The paper's results can be reproduced with the following guideline.

### Requirements
To use the code, you require the following tools as prerequisites:
* Python 3.7
* git
    
We recommend Anaconda or Miniconda, the latter being a minimal (but sufficient) version of the Anaconda distribution. The following instructions will be based on Miniconda. If you use another Python environment, the installation routine must be adapted.

### Quickstart
After the installation of Anaconda, open a terminal (on Windows Anaconda Prompt) and create a new environment by typing:


##### Clone repository and install requirements
```
git clone https://github.com/osaukh/pollenpub
cd code/
conda env create -f environment.yml
```

Note: The following commands are to be run from the `code/` directory.

##### Download weights and pollen images
```
python utils/download.py -f weights.zip data.zip
```

##### Run the test
```
python test.py --weights_path ../weights/pollen_20190526.pth --model_def config/yolov3-pollen.cfg --data_config config/test_20190523.data
```

##### Run object detection on any image
Change the `--image_folder` argument to an image directory which contains pollen images and detect pollen images with the following command:

```
python detect.py --image_folder ../data/pollen_20190523/layers/ --output_folder ../tmp/output/ 
```

### Training
Training on the provided dataset can be done by issuing the following command

```
python train.py --name fold0 --epochs=60 --model_def config/yolov3-pollen.cfg --data_config config/train_20190526fold0.data --pretrained_weights ../weights/darknet53.conv.74
```

#### Prepare new pollen training sets
The create_folds.py script can be used to prepare the image folders to be used with the training script.
It creates one/multiple text files which can be used for training/testing. Each file contains the name of the images belonging to the train/val set. The script takes into account that each sample consists of multiple depth layers since it is important that all depth layers of a sample are either in train or val set. This avoids information leakage between test and val,

Note: The following commands need only to be exectuted if you want to use a new labeled image folder for training. The files produced by these commands are already in [./config/](./config/).

After preparing the data you can run the training procedure as described before but with updated `--data_config` parameter.

##### Create training an validation set
```
python create_folds.py -f ../data/pollen_20190526/ -o ../data/pollen_20190526/ -K 5 -n train_20190526
```

##### Create test set
```
python create_folds.py -f ../data/pollen_20190523/ -o ../data/pollen_20190523/ -n test_20190523
```


### Credit

#### PyTorch-YOLOv3 repository
Thanks to [Erik Linder-Norén](https://github.com/eriklindernoren/) who open sourced his [YOLOv3 code](https://github.com/eriklindernoren/PyTorch-YOLOv3) on which this implementation is based.

#### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
