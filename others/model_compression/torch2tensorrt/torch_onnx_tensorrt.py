import tensorrt
import torch
from torch2trt import torch2trt
import onnx
# import onnxruntime as rt
import cv2
import numpy as np
import onnx_tensorrt.backend as backend

print(tensorrt.__version__)

def get_img_np_nchw(filename):
    image = cv2.imread(filename)
    image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_cv = cv2.resize(image_cv, (640, 640))
    miu = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229,0.224,0.225])
    img_np = np.array(image_cv, dtype=float) / 255.
    r = (img_np[:, :, 0] - miu[0]) / std[0]
    g = (img_np[:, :, 1] - miu[1]) / std[1]
    b = (img_np[:, :, 2] - miu[2]) / std[2]
    img_np_t = np.array([r, g, b])
    img_np_nchw = np.expand_dims(img_np_t, axis=0)
    return np.asarray(img_np_nchw,dtype=np.float32)



def torch2trt_lg():
    model_path = 'yolov5_saved_from_torch.pth'
    filename = '/media/dell/data/voc/VOCdevkit/VOC2007/trainval/JPEGImages/000005.jpg'
    imgdata = torch.from_numpy(get_img_np_nchw(filename)).cuda()
    model = torch.load(model_path).cuda()
    y = model(imgdata)

    model_trt = torch2trt(model, [imgdata],
                          input_names=['img'],
                          output_names=["f1", 'f2', 'f3'],
                          use_onnx=True)
    y_trt = model_trt(imgdata)

    print(torch.max(torch.abs(y - y_trt)))


def torch2onnx():
    model = 'yolov5_saved_from_torch.pth'
    filename = '/media/dell/data/voc/VOCdevkit/VOC2007/trainval/JPEGImages/000005.jpg'
    imgdata = torch.from_numpy(get_img_np_nchw(filename)).cuda()
    model = torch.load(model)
    out = model.forward(imgdata)
    torch.onnx.export(model, args=imgdata, f="yolov5_generated_from_torch.onnx",
                      export_params=True,
                      verbose=True,
                      input_names=['img'],
                      output_names=["f1", 'f2', 'f3'],
                      opset_version=11,enable_onnx_checker=True)
    a=0

def onnx2trt_with_code():  # failed
    # with code
    model = onnx.load("yolov5_generated_from_torch.onnx")
    engine = backend.prepare(model, device='CUDA:0')
    input_data = np.random.random(size=(4, 3, 640, 640)).astype(np.float32)
    output_data = engine.run(input_data)[0]
    print(output_data)
    print(output_data.shape)


def onnx2trt_with_trt():  # failed
    trt_path = 'yolov5.trt'
    onnx_path = 'yolov5.onnx'


def onnx_runtime():
    # imgdata = np.asarray(np.random.rand(1, 3, 640, 640), dtype=np.float32)
    filename = '/media/dell/data/voc/VOCdevkit/VOC2007/trainval/JPEGImages/000005.jpg'
    imgdata = get_img_np_nchw(filename)
    sess = rt.InferenceSession('yolov5.onnx')
    input_name = sess.get_inputs()[0].name
    output_name = [sess.get_outputs()[0].name, sess.get_outputs()[1].name, sess.get_outputs()[2].name]
    pred_onnx = sess.run(output_name, {input_name: imgdata})
    print("outputs:")


if __name__ == '__main__':
    torch2trt_lg()
    # torch2onnx()
    # onnx_runtime()
    onnx2trt_with_code()
