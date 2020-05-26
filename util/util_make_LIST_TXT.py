import glob
import os


def make_list(img_path, lab_path):
    dir_list = []
    path_list = []
    for path in img_path:
        for ex_name in expand_name:
            path_list += glob.glob(path + '/*' + ex_name)

        for path in path_list:
            file_name = os.path.basename(path)
            path_2 = os.path.join(lab_path, file_name)
            front = path_2.split('.')[0]
            path_2 = front + '.txt'

            if limit_fun(path, path_2):
                dir_list.append([file_name, path, path_2])
                print('adding:', dir_list[-1])

    return dir_list


def limit_fun(path, path_2):
    if not os.path.isfile(path_2):
        return False
    lines = open(path_2).readlines()
    if lines == []:
        return False
    else:
    # for line in lines:
    #     name_dict = {'0': 'ignored regions', '1': 'pedestrian', '2': 'people',
    #                  '3': 'bicycle', '4': 'car', '5': 'van', '6': 'truck',
    #                  '7': 'tricycle', '8': 'awning-tricycle', '9': 'bus',
    #                  '10': 'motor', '11': 'others'}
    #     tmps = line.strip().split(',')
    #     realname = name_dict[tmps[5]]
    #     if realname in ['car', 'van', 'truck', 'bus']:
    #         return True
        return True
    return False


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
    # pathes = ['E:/datasets/VisDrone2019/VisDrone2019-DET-train/']
    # img_path = ['E:/datasets/VisDrone2019/VisDrone2019-DET-train/images']
    # lab_path = 'E:/datasets/VisDrone2019/VisDrone2019-DET-train/annotations'
    img_path = ['F:\Projects\\auto_Airplane\TS02\\20191220_1526019_20/']
    lab_path = 'F:\Projects\\auto_Airplane\TS02\\20191220_1526019_20_refined/'
    expand_name = ['.jpg', '.png']
    save_path = 'util_tmp/make_list.txt'
    datalist = make_list(img_path, lab_path)
    _wrte_dataset_txt(datalist, save_path)
