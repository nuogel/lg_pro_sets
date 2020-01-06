"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""

import _pickle as cPickle
import logging
import numpy as np
import shutil
import os
import xml.etree.ElementTree as ET
from util.util_filename_to_list import just_get_basename

logger = logging.getLogger(__name__)
object_size_clip = [0, 7000, 90000, 1e9]


def parse_rec(filename, area_size=[0, 1000]):
    """Parse a Labels of xml file."""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        # obj_struct['pose'] = obj.find('pose').text
        # obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = 0
        bbox = obj.find('bndbox')
        x1, y1, x2, y2 = int(float(bbox.find('xmin').text)), int(float(bbox.find('ymin').text)), \
                         int(float(bbox.find('xmax').text)), int(float(bbox.find('ymax').text))
        area = (x2 - x1) * (y2 - y1)
        if not area_size[0] < area < area_size[1]:
            continue
        obj_struct['bbox'] = [x1, y1, x2, y2]
        objects.append(obj_struct)
    return objects


def convert_txt2clas(basename_file, pre_lab_dir, area_size):
    """Convert predicted labels to class labels.like: Car file_name labels..."""
    if os.path.isdir('./cache/prepath/'):
        shutil.rmtree('./cache/prepath/')
    os.mkdir('./cache/prepath/')
    cls_names = []
    # if not os.path.isfile(pre_cls_path):
    #     os.mknod(pre_cls_path)
    for basename in open(basename_file, 'r').readlines():
        basename = basename.split(';')[0].strip()
        print("converting txt to classes:", basename)

        for line in open(os.path.join(pre_lab_dir, basename + '.txt'), 'r').readlines():
            tmp = line.split()
            [x1, y1, x2, y2] = tmp[2:]
            area = (int(x2) - int(x1)) * (int(y2) - int(y1))
            if not area_size[0] < area < area_size[1]:
                continue
            pre_cls_path = './cache/prepath/pre_{}.txt'.format(tmp[0])
            with open(pre_cls_path, 'a') as f:
                f.write('{} {} {} {} {} {}\n'.format(basename, tmp[1], tmp[2], tmp[3], tmp[4], tmp[5]))
                if tmp[0] not in cls_names:
                    cls_names.append(tmp[0])
    return cls_names


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(imagesetfile,
             classname,
             gt_path,
             area_size,
             reconvert_labels=False,
             cachedir='./cache/',
             ovthresh=0.5,
             use_07_metric=False,
             ):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    imageset = os.path.splitext(os.path.basename(imagesetfile))[0]
    cachefile = os.path.join(cachedir, imageset + '_annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if (not os.path.isfile(cachefile)) or reconvert_labels:
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            # print('converting labels to  pickle:', imagename)
            xml_f = gt_path.format(imagename)  # imagename[:-2]
            recs[imagename] = parse_rec(xml_f, area_size)
            print('Reading annotation for {:d}/{:d}'.format(
                i + 1, len(imagenames)))
        # save
        logger.info('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detpath = os.path.join('./cache/prepath//pre_{}.txt')
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return prec, rec, ap


def mAP_voc(pre_path, gt_path, reconvert_labels=False, object_size_level=3):
    base_name = "cache/basename.txt"
    if object_size_level not in [-1, 1, 2, 3]:
        print('wrong size level.')
        return 0
    if object_size_level == -1:
        area_size = [object_size_clip[0], object_size_clip[-1]]
    else:
        area_size = [object_size_clip[object_size_level - 1], object_size_clip[object_size_level]]

    if reconvert_labels:
        classnames = convert_txt2clas(base_name, pre_path, area_size)

    else:
        classnames = ['Car', 'Pedestrian']
    mAP = 0.
    result_txt = "********** results **********\n"
    for classname in classnames:
        prec, rec, ap = voc_eval(base_name, classname, gt_path, area_size, reconvert_labels)
        result_txt += classname + "'s AP:" + str(ap) + '\n'
        mAP += ap
    result_txt += "---------\nmAP:" + str(mAP / (len(classnames) + 1e-9))
    print(result_txt)
    result_f = open('cache/results.txt', 'w')
    result_f.write(result_txt)
    result_f.close()
    return mAP


if __name__ == '__main__':
    # pre_path = "E:\LG\programs\lg_pro_sets//tmp//kitti_level5//"
    # pre_path = "E://LG//programs//eva_sys//datasets//result//"
    pre_path = 'E:/LG/GitHub/lg_pro_sets/tmp/predicted_labels/'


    # gt_path = "E://LG//programs//lg_pro_sets//datasets//Annotations_kitti//training//{}.xml"
    # gt_path = 'E:/datasets/Car/VOC_Car/labels//{}.xml'
    gt_path = 'E:/datasets/VOCdevkit/Annotations/{}.xml'
    # gt_path = 'E:/datasets/kitti/training/labels_xml/{}.xml'
    just_get_basename(pre_path)
    reconvert_labels = True
    object_size_level = -1
    mAP_voc(pre_path, gt_path, reconvert_labels, object_size_level)
