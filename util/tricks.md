## import 
mod = importlib.import_module('numpy')
means: import mumpy
## 多线程处理数据
```
from concurrent.futures import ProcessPoolExecutor
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
