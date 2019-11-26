from dataloader import loader_asr, loader_img, loader_sr

DataLoaderDict = {"OBD": loader_img.DataLoader,
                  "ASR": loader_asr.DataLoader,
                  "SR": loader_sr.DataLoader,

                  }
