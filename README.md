#### V1.0.0_STEADY : luogeng 
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

