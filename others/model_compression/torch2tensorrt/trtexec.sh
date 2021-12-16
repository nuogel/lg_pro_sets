/media/dell/data/installpkgs/TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn7.6/TensorRT-7.0.0.11/targets/x86_64-linux-gnu/bin/trtexec \
--explicitBatch \
--onnx=/home/dell/lg/code/lg_pro_sets/others/model_compression/torch2tensorrt/tmp/yolov5_with_model.pth.onnx \
--saveEngine=yolov5_with_model.trt_engine \
#--fp16 \
--workspace=10240 \
--verbose
