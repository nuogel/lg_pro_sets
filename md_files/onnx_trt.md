onnx 初探：https://mp.weixin.qq.com/s/H1tDcmrg0vTcSw9PgpgIIQ
![img.png](img.png)
![img_2.png](img_2.png)



# experiments:
test report:

times-type | result(ms/img) |-
---|---|---
100 times of pytorch-cpu | 128.8730549812317  |-
100 times of onnx-cpu |     83.4130311012268 |-
100 times of pytorch-gpu |  26.613991260528564  |-
100 times of tensorRt-gpu |  0.25597095489501953  |-