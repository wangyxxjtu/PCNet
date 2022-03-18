# Generating Superpixels for High-resolution Images with Decoupled Patch Calibration 

This is is a PyTorch implementation of the superpixel segmentation network introduced in this paper (2021):

[Generating Superpixels for High-resolution Images with Decoupled Patch Calibration](https://arxiv.org/pdf/2108.08607.pdf)

## Introduction
The Illustration of PCNet:

<img src="https://github.com/wangyxxjtu/PCNet/master/workflow-crop.pdf" width="845" alt="workflow" />

The visual comparison of our PCNet and the SOTA methods:

<img src="https://github.com/wangyxxjtu/PCNet/blob/master/argue-crop.pdf" width="845" alt="workflow" />

By merging superpixels, some object proposals could be generated:

<img src="https://github.com/wangyxxjtu/PCNet/blob/master/framework/object_proposal.png" width="845" alt="workflow" />

## Prerequisites
The training code was mainly developed and tested with python 2.7, PyTorch 0.4.1, CUDA 9, and Ubuntu 16.04.

During test, we make use of the component connection method in [SSN](https://github.com/NVlabs/ssn_superpixels) to enforce the connectivity 
in superpixels. The code has been included in ```/third_paty/cython```. To compile it:
 ```
cd third_party/cython/
python setup.py install --user
cd ../..
```
## Demo
Quick taste! Specify the image path and use the [pretrained model](https://drive.google.com/file/d/1WDcU7Oa5U4p37-prrA8f51IM3ycrtuCp/view?usp=sharing) to generate superpixels for an image
```
python run_demo.py --image=PATH_TO_AN_IMAGE --output=./demo 
```
The results will be generate in a new folder under ```/demo``` called ```spixel_viz```.

 
## Data preparation 
For high-resolution superpixel segmentation, download the [FaceHuman dataset](https://drive.google.com/file/d/1819lO7P56-FIJ6Md3gszwoSnNZf9llGl/view?usp=sharing)

To test our model on low-resolution datasets , please first download the data from the original [BSDS500 dataset](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_full.tgz), 
and extract it to  ```<BSDS_DIR>```. Then, run 
```
cd data_preprocessing
python pre_process_bsd500.py --dataset=<BSDS_DIR> --dump_root=<DUMP_DIR>
python pre_process_bsd500_ori_sz.py --dataset=<BSDS_DIR> --dump_root=<DUMP_DIR>
cd ..
```
The code will generate three folders under the ```<DUMP_DIR>```, named as ```/train```, ```/val```, and ```/test```, and three ```.txt``` files 
record the absolute path of the images, named as ```train.txt```, ```val.txt```, and ```test.txt```.


## Training
Once the data is prepared, we should be able to train the model by running the following command:
```
python main.py --data=<DATA_DIR> --savepath=<PATH_TO_SAVE_CKPT> --workers 4 --input_img_height 208 --input_img_width 208 --print_freq 20 --gpu 0 --batch-size 16  --suffix '_myTrain' 
```
If you want to continue training from a ckpt, just add --pretrained=<PATH_TO_CKPT>. You can specify the training config in the 'train.sh' script.

The training log can be viewed from the `tensorboard` session by running
```
tensorboard --logdir=<CKPT_LOG_DIR> --port=8888
```

If everything is set up properly, reasonable segmentation should be observed after 10 epochs.

## Testing
We provide test code to generate: 1) superpixel visualization and 2) the```.csv``` files  for evaluation. 

To test on Face-Human dataset, run
```
python run_infer_face.py --data_dir=<DUMP_DIR> --output=<TEST_OUTPUT_DIR> --pretrained=<PATH_TO_THE_CKPT>
```

To test on BSDS500, run
```
python run_infer_bsds.py --data_dir=<DUMP_DIR> --output=<TEST_OUTPUT_DIR> --pretrained=<PATH_TO_THE_CKPT>
```

## Evaluation
We use the code from [superpixel benchmark](https://github.com/davidstutz/superpixel-benchmark) for superpixel evaluation. 
A detailed  [instruction](https://github.com/davidstutz/superpixel-benchmark/blob/master/docs/BUILDING.md) is available in the repository, please
 
(1) download the code and build it accordingly;

(2) edit the variables ```$SUPERPIXELS```, ```IMG_PATH``` and ```GT_PATH``` in ```/eval_spixel/my_eval.sh```,
example:

```
IMG_PATH='/home/name/superpixel/PCNet/FaceHuman/test'
GT_PATH='/home/name/superpixel/PCNet/FaceHuman/test/map_csv'

../../bin_eval_summary_cli /home/name/superpixel/PCNet/eval/test_multiscale_enforce_connect/SPixelNet_nSpixel_${SUPERPIXEL}/map_csv $IMG_PATH $GT_PATH

```

(3)run 
```
cp /eval_spixel/my_eval.sh <path/to/the/benchmark>/examples/bash/
cd  <path/to/the/benchmark>/examples/
bash my_eval.sh
```

(4) run 
 ```
cp ./eval_spixel/my_eval.sh <path/to/the/benchmark>/examples/bash/
cd  <path/to/the/benchmark>/examples/

#the results will be saved to: /home/name/superpixel/PCNet/eval/test_multiscale_enforce_connect/SPixelNet_nSpixel_54/map_csv/
bash my_eval.sh
 ```
several files should be generated in the ```map_csv``` folders in the corresponding test outputs including summary.txt, result.txt etc;

(5) cd PCNet/eval_spixel
```
python plot_benchmark_curve.py --path '/home/name/superpixel/PCNet/eval/test_multiscale_enforce_connect/' #will generate the similar curves in the paper
```

## Citation
If you use our code, please cite our work:
``` bash
{

}
```

## Acknowledgement
This code is built on the top of SCN: https://github.com/fuy34/superpixel_fcn Thank the authors' contribution. 
