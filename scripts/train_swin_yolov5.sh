python train.py --type obd --model swin_yolov5 --ep 30 --cp 1  --bz 4 --lr 0.001 --nw 8 --ema 1 --autoamp 1 --pt 'saved/swin_yolov5_448x448_voc_74.5/672x672-0.766.pkl'

#python train.py --type OBD --ep 801 --model yolov5  --cp 'saved/checkpoint/yolov5_voc_68.pkl'  --bz 16 --nw 8 --ema 1 --autoamp 1 --to 1

