from dataloader import loader_asr, loader_img, Loader_SR_DN

DataLoaderDict = {"OBD": loader_img.DataLoader,
                  "ASR": loader_asr.DataLoader,
                  "SR_DN": Loader_SR_DN.DataLoader,

                  }
