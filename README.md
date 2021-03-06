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
    SRCNN
    FSRCNN
    ESPCN
    VDSR
    EDSR
    RDN (Residual Dense Network)
    RCAN
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
-------------------------
Notice:
if python-Levenshtein failed，then try Pipy, python-Levenshtein-wheels.


#### Training Dataset
you can use any data sets in shape of [images, labels], labels can be **.xml or **.txt
```
OBD:KITTI, VOC, COCO
ASR: SPEECH
SRDN: youku dataset
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

networks | input size |  F1-SCORE |mAP| weight size| PS
 --- | --- | --- |  --- |---|---
yolov2|512x768|0.86|X|58.5 M|used 16 Anchors.
yolov3|384x960|0.9|X|136 M|收敛快，效果好
yolov3_tiny | 512x768| 0.857 |0.76571836| 33 M|
yolov3_tiny_squeezenet | 384x960 | 0.844 |X|5.85 M|收敛快，效果好
yolov3_tiny_mobilenet|512x768|0.836|X|3.37 M|
yolov3_tiny_shufflenet|512x768|0.790|0.672|686 KB|
refinedet | 512x768 | 0.91|X|129 M|收敛快，效果好
efficientdet_b0|512x768|0.9|X|42.7M|收敛快，效果好
ssd|512x768|0.8904|X|94.7 M|收敛慢，效果好

以上分数是在 SCORE_THRESH: 0.7 下得到的，以yolov3_tiny为例: 
SCORE_THRESH: 0.7 :
FSCORE:0.857 ;AP: 0.76571836; precision 0.96, recall 0.77。
如果设为SCORE_THRESH 0.6
FSCORE:0.866 ;AP: 0.80; precision 0.93, recall 0.806
如果设为SCORE_THRESH 0.55
FSCORE:0.87006 ;AP: 0.8105; precision 0.92127, recall 0.82423

score thresh|f1-score|AP|precision|recall
---|---|---|---|---
0.7|0.857|0.7657|0.96,|0.77
0.6|0.866|0.80|0.93,|0.806
0.55|0.870|0.8105|0.921|0.824
0.5|0.871|0.822|0.905|0.840

=================================================

ASR:

networks | WRR |weight size| PS
 --- | --- | --- |  --- 
CTC         |91.0|154 M|XXX
Seq2Seq     |95.0|58.5 M|XXX
Transformer |0.1|58.5 M|XXX

==================================================

SR: 

dataset:youku

networks | PSNR |weight size| PS
 --- | --- | --- |  --- 
SRCNN|X|X|XXX
FSRCNN|X|X|XXX
VDSR |33.66|X|XXX
EDSR |30.97|X|XXX
RCAN |35.73| XX|XXX 
CBDNET |XX| XX|XXX 
RDN |35.5| XX|XXX 


