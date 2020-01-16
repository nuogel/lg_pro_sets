#### V1.1.0 STEADY : luogeng->master 
### LuoGeng's Programs Set (Forbid Copying)
This is a PyTorch implementation of ASR / OBD / SR /DENORSE /TRACK
#### ASR:
    CTC
    Seq2Seq
    RNN
    LSTM
    Transformer
#### OBD/OBC:
    YOLOv2
    YOLOV3
    YoloV3_Tiny
    YoloV3_Tiny_SqueezeNet
    YoloV3_Tiny_MobileNet
    YoloV3_Tiny_ShuffleNet
    YoloV3_MobileNet
    FCOS
    RefineDet
    MobileNetV3
#### SR:
    EDSR
#### DENORSE:
    DnCnn
    CBDNet
    
#### TRACK:
    KCF


#### Runtime environment
you need to install all the environment before you enjoy this code.
```
pytorch
numpy
pandas
torch
torchvision
...
conda install -c conda-forge imgaug 

```
#### Training Dataset
you can use any data sets in shape of [images, labels], labels can be **.xml or **.txt
```
KITTI, VOC, COCO, SPEACH ...
```

#### Train The Model
```
python train.py --yml_path xxx  --checkpoint xxx --lr xxx
```
#### Test The Model
```
python test.py --yml_path xxx --checkpoint xxx
```

#### The results 
OBD:
The result of F1-score of Car in KITTI DATA SET.

networks | input size |  F1-SCORE |weight size| PS
 --- | --- | --- |  --- |---
yolov2|512x768|0.86|58.5 M|used 16 Anchors.
yolov3|384x960|0.9|136 M|收敛快，效果好
yolov3_tiny | 512x768| 0.857 | 33 M|收敛快，效果好
yolov3_tiny_squeezenet | 384x960 | 0.844 |5.85 M|收敛快，效果好
yolov3_tiny_mobilenet|512x768|0.836|3.37 M|
yolov3_tiny_shufflenet|512x768|0.726|686 KB|
refinedet | 512x768 | 0.91|129 M|收敛快，效果好
efficientdet_b0|512x768|0.9|42.7M|收敛快，效果好
ssd|512x768|0.8904|94.7 M|收敛慢，效果好

=================================================

ASR:

networks | WER |weight size| PS
 --- | --- | --- |  --- 
CTC         |0.1|154 M|XXX
Seq2Seq     |0.1|58.5 M|XXX
Transformer |0.1|58.5 M|XXX

