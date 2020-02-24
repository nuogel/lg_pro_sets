from dataloader import loader_asr, loader_img, loader_SR_DN

DataLoaderDict = {"OBD": loader_img.DataLoader,
                  "ASR": loader_asr.DataLoader,
                  "SR_DN": loader_SR_DN.DataLoader,

                  }
