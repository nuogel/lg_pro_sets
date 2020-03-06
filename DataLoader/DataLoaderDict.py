from DataLoader import Loader_ASR, Loader_OBD, Loader_SRDN

DataLoaderDict = {"OBD": Loader_OBD.DataLoader,
                  "ASR": Loader_ASR.DataLoader,
                  "SRDN": Loader_SRDN.DataLoader,

                  }
