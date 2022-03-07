python train.py --type obd --model swin_yolov5 --ep 50 --cp 0  --bz 12 --lr 0.001 --nw 8 --ema 1 --autoamp 1 --pt 'saved/checkpoint/sw-yolov5-coco-pretrained.pkl'

#python train.py --type OBD --ep 801 --model yolov5  --cp 'saved/checkpoint/yolov5_voc_68.pkl'  --bz 16 --nw 8 --ema 1 --autoamp 1 --to 1