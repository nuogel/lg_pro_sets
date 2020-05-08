import glob
import os


def make_list(pathes):
    dir_list = []
    for path in pathes:
        path_list = glob.glob(path + '*/*.jpg')
        for path in path_list:
            file_name = os.path.basename(path)
            path_2 = path.replace('images', 'annotations')
            path_2 = path_2.replace('jpg', 'txt')
            # path_2 = path_2.replace('img', 'gt_img')

            if not os.path.isfile(path_2):
                continue
            dir_list.append([file_name, path, path_2])

    return dir_list


def _wrte_dataset_txt(dataset, save_path):
    data_set_txt = ''
    for i in dataset:
        data_set_txt += str(i[0]) + ';' + str(i[1]) + ';' + str(i[2]) + '\n'  # '\n'  # +
    f = open(save_path, 'w')
    f.write(data_set_txt)
    f.close()


if __name__ == '__main__':
    # pathes = ['D:/datasets/CCPD2019/ccpd_challenge/']
    # pathes = ['F:/datasets/SR/REDS4/train_sharp_part/']
    # pathes = ['F:/LG/OCR/PAN.pytorch-master/dadaset/wxf_ocr_data/']
    # pathes = ['E:/datasets/youku/youku_00200_00249_h_GT/']
    pathes = ['E:/datasets/VisDrone2019/VisDrone2019-DET-train/']
    save_path = 'util_tmp/make_list.txt'
    datalist = make_list(pathes)
    _wrte_dataset_txt(datalist, save_path)
