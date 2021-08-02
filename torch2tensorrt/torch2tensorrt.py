import tensorrt
import torch
import onnx
import onnxruntime as rt

import numpy as np
import onnx_tensorrt.backend as backend

print(tensorrt.__version__)


def torch2onnx():
    model = './yolov5.pth'
    dummy_input = torch.randn([1, 3, 640, 640])
    model = torch.load(model)
    out = model.forward(dummy_input)
    torch.onnx.export(model, args=dummy_input, f="yolov5.onnx",
                      export_params=True,
                      verbose=True,
                      input_names=['label'],
                      output_names=["synthesized"],
                      opset_version=11)


def onnx2trt_with_code():  # failed
    # with code
    model = onnx.load("yolov5.onnx")
    engine = backend.prepare(model, device='CUDA:0')
    input_data = np.random.random(size=(4, 3, 640, 640)).astype(np.float32)
    output_data = engine.run(input_data)[0]
    print(output_data)
    print(output_data.shape)


def onnx2trt_with_trt():  # failed
    trt_path='yolov5.trt'
    onnx_path='yolov5.onnx'


def onnx_runtime():
    imgdata = np.randn([1, 3, 640, 640])
    sess = rt.InferenceSession('test.onnx')
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    pred_onnx = sess.run([output_name], {input_name: imgdata})
    print("outputs:")
    print(np.array(pred_onnx))

if __name__ == '__main__':
    torch2onnx()