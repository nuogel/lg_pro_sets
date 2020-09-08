## import 
mod = importlib.import_module('numpy')
means: import mumpy

## 多线程处理数据
```
from concurrent.futures import ProcessPoolExecutor
from functools import partial

executor = ProcessPoolExecutor(max_workers=num_workers)
executor.submit(partial(fun, out_dir, index, wav_path, text)))

def fun(out_dir, index, wav_path, text):
    ...
```
## 语音特征
    import audio

    wav = audio.load_wav(wav_path)

    if hparams.rescaling:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

# PIP tricks
##  linux pip install
```
pip show numpy

```

the default packages location of pip install is '/home/lg/.local/lib/python3.6/'


##  pip commands

pip install opencv-python
pip install torch torchvision

when pip reinstall the packages that already exist, 'python -m pip install --upgrade pip' might solve that problem.
## pip  source
```
Linux下，修改 ~/.pip/pip.conf
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host=mirrors.aliyun.com

windows下，在user目录中创建一个pip目录，如：C:\Users\xx\pip，新建文件pip.ini。内容同上。
```

##  linux pip install
```
pip show numpy

```

the default packages location of pip install is '/home/lg/.local/lib/python3.6/'


##  pip install opencv3

pip install opencv

##  delete nvidia-driver
```
sudo apt-get --purge remove nvidia*
sudo apt autoremove

To remove NVIDIA Drivers:
$ sudo apt-get --purge remove "*nvidia*"

To remove CUDA Toolkit:
$ sudo apt-get --purge remove "*cublas*" "cuda*"
```


#Torch Tricks

## sort() and natsort()
```
from natsort import natsorted
a = ['1.mp4', '3.mp4', '10.mp4', '2.mp4']
b = natsorted(a)
print(b)

```


