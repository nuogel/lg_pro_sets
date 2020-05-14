# -*- coding: utf-8 -*-
"""
eva_sys - 当前项目的名称。
util_show_img - 读取图片数据，对应标签数据，将标注框画到图片上，显示出来

"""
import cv2
import os
import xml.etree.ElementTree as ET


def get_ALL_File(dir_path, seq='.png'):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(seq):
                filename = os.path.join(root, file)
                file_list.append(filename)

    return file_list


def _read_line(path):
    objs = []
    if os.path.basename(path).split('.')[-1] == 'txt':
        file_open = open(path, 'r')
        for line in file_open.readlines():
            if ";" in line.strip():
                tmps = []
                tmps.append(line.strip().split(";")[0])
                tmps.extend(line.strip().split(";")[1].split(' '))
            else:
                tmps = line.strip().split(' ')
            # print(len(tmps))
            # print(tmps)
            name = tmps[0]
            if len(tmps) == 5:
                box_x1 = float(tmps[1])
                box_y1 = float(tmps[2])
                box_x2 = float(tmps[3])
                box_y2 = float(tmps[4])
            else:
                box_x1 = float(tmps[4])
                box_y1 = float(tmps[5])
                box_x2 = float(tmps[6])
                box_y2 = float(tmps[7])
            # if name in CLASS:
            objs.append([name, str(box_x1), str(box_y1), str(box_x2), str(box_y2)])
    elif os.path.basename(path).split('.')[-1] == 'xml':

        tree = ET.parse(path)
        # 　label_path = fpath.replace("images", "labels")
        # im_file = in_file[:-4].replace("labels", "images") + ".jpg"
        # img = cv2.imread(im_file)
        root = tree.getroot()
        # bndboxlist = []
        for object in root.findall('object'):  # 找到root节点下的所有country节点
            bndbox = object.find('bndbox')  # 子节点下节点rank的值
            name = object.find('name').text
            xmin = float(bndbox.find('xmin').text)
            xmax = float(bndbox.find('xmax').text)
            ymin = float(bndbox.find('ymin').text)
            ymax = float(bndbox.find('ymax').text)
            # if name in CLASS:
            objs.append([name, str(xmin), str(ymin), str(xmax), str(ymax)])

    return objs


def _write_line(before_file, new_target, after_file):
    if os.path.basename(before_file).split('.')[-1] == 'txt':
        lines = ''
        for index in range(len(new_target)):
            new_xmin = new_target[index][0]
            new_ymin = new_target[index][1]
            new_xmax = new_target[index][2]
            new_ymax = new_target[index][3]
            name = new_target[index][4]
            line = name + " " + '%.4f' % new_xmin + " " + '%.4f' % new_ymin + " " + '%.4f' % new_xmax + " " + '%.4f' % new_ymax + "\n"
            lines += line
        with open(after_file, 'w') as f:
            f.write(lines)
    elif os.path.basename(before_file).split('.')[-1] == 'xml':
        tree = ET.parse(before_file)
        elem = tree.find('filename')
        fpath, fname = os.path.split(after_file)
        elem.text = (fname[:-4] + '.jpg')
        # size = tree.find('size')
        # height, width, channel = img.shape
        # h = size.find('height')
        # h.text = str(height)
        # w = size.find('width')
        # w.text = str(width)
        # depth = size.find('depth')
        # depth.text = str(channel)

        xmlroot = tree.getroot()
        index = 0
        # print(len( xmlroot.findall('object')))

        for object in xmlroot.findall('object'):  # 找到root节点下的所有country节点
            bndbox = object.find('bndbox')  # 子节点下节点rank的值

            new_xmin = new_target[index][1]
            new_ymin = new_target[index][2]
            new_xmax = new_target[index][3]
            new_ymax = new_target[index][4]

            xmin = bndbox.find('xmin')
            xmin.text = new_xmin
            ymin = bndbox.find('ymin')
            ymin.text = new_ymin
            xmax = bndbox.find('xmax')
            xmax.text = new_xmax
            ymax = bndbox.find('ymax')
            ymax.text = new_ymax

            index = index + 1

        tree.write(after_file)


def _read_datas(im_file, lab_file):
    img = cv2.imread(im_file)
    if img is not None:
        image_size = img.shape[:-1]  # the last pic as the shape of a batch.
    # images.append(img)
    else:
        print('imread img is NONE.')

    label = _read_line(lab_file)

    if label is None:
        print("label error")

    return img, label


def _show_img(images, labels, show_img=True, show_time=30000, save_img=False):
    if labels:
        for _, label in enumerate(labels):
            if len(label) == 5:  # in shape of [class, x1, y1, x2, y2]
                class_out, box = label[0], label[1:5]

            elif len(label) == 2:  # [class, bbox[x1,y1,x2,y2]]
                class_out = label[0]
                box = label[1]
            else:
                print('error: util_show_img-->no such a label shape')
                continue
            xmin = int(float(box[0]))
            ymin = int(float(box[1]))
            xmax = int(float(box[2]))
            ymax = int(float(box[3]))
            # text = class_out+"||"+ " ".join([str(i) for i in box])
            text = class_out
            cv2.rectangle(images, (xmin, ymin), (xmax, ymax), (255, 0, 0))
            cv2.putText(images, text, (xmin, ymin), 1, 1, (0, 0, 255))

    if show_img:
        cv2.imshow('img', images)
        cv2.waitKey(show_time)
    if save_img:
        cv2.imwrite("show.png", images)


def main():
    print_path = True
    # path = 'E:/datasets/Car/COCO_Car'
    # path = 'E:/datasets/BDD100k/'
    # path ='E:/datasets/OpenImage_Car'
    path = 'E:/datasets/VisDrone2019/VisDrone2019-VID-train/'
    # im_file = os.path.join(path, "images", "1478019971185917857.jpg")
    # label_file = os.path.join(path, "labels", "1478019971185917857.xml")
    # img, label = _read_datas(im_file, label_file)

    # _show_img(img, label)
    local_xml_files = get_ALL_File(path, ".xml")

    img_files = get_ALL_File(path, ".jpg")
    # print(len(local_img_files))
    png_files = get_ALL_File(path, ".png")
    # print(len(png_files))
    local_img_files = img_files + png_files
    local_img_files.sort()
    local_xml_files.sort()
    lab_num = len(local_xml_files)
    img_num = len(local_img_files)
    print(lab_num, img_num)

    for index in range(min(img_num, lab_num)):
        label_file = local_xml_files[index]
        im_file = local_img_files[index]
        if print_path:
            print(im_file, '==>>>', label_file)
        img, label = _read_datas(im_file, label_file)
        _show_img(img, label)


if __name__ == '__main__':
    main()
