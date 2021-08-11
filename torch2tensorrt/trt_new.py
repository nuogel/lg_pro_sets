import torch
from torch2trt import torch2trt

# create some regular pytorch model...

model = torch.load('./yolov5.pth').cuda()
# create example data
x = torch.ones((1, 3, 224, 224)).cuda()

y = model(x)

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])

y = model(x)
y_trt = model_trt(x)

# check the output against PyTorch
print(torch.max(torch.abs(y - y_trt)))


torch.save(model_trt.state_dict(), 'alexnet_trt.pth')

from torch2trt import TRTModule

model_trt = TRTModule()

model_trt.load_state_dict(torch.load('alexnet_trt.pth'))

y_trt = model_trt(x)
print(torch.max(torch.abs(y - y_trt)))
