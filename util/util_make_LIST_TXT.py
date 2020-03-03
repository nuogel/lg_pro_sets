import glob
import os


def make_list(pathes):
    dir_list = []
    for path in pathes:
        path_list = glob.glob(path + '/*.jpg')
        for path in path_list:
            file_name = os.path.basename(path)
            # path_2 = path.replace('h_GT', 'l')
            # if not os.path.isfile(path_2):
            #     continue
            dir_list.append([file_name, 'None', path])

    return dir_list


def _wrte_dataset_txt(dataset, save_path):
    data_set_txt = ''
    for i in dataset:
        data_set_txt += str(i[0]) + ';' + str(i[1]) + ';' + str(i[2]) + '\n'
    f = open(save_path, 'w')
    f.write(data_set_txt)
    f.close()


if __name__ == '__main__':
    pathes = ['D:/datasets/CCPD2019/ccpd_base/']
    save_path = 'util_tmp/make_list.txt'
    datalist = make_list(pathes)
    _wrte_dataset_txt(datalist, save_path)
