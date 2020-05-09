import cv2


def img_cut(img, gap=200, target_size=(512, 768)):
    cv2.imshow('img_', cv2.resize(img, None, fx=0.8, fy=0.8))
    cv2.waitKey()
    [h, w, c] = img.shape
    t_w = target_size[1]
    t_h = target_size[0]

    H_OK = False
    img_cuts = []
    cut_ha = 0
    cut_hb = t_h
    while 1:
        cut_wa = 0
        cut_wb = t_w
        W_OK =False
        while 1:
            img_cut = img[cut_ha:cut_hb, cut_wa:cut_wb, :]
            img_cuts.append(img_cut)
            cv2.imshow('img', img_cut)
            cv2.waitKey()
            if W_OK:
                break
            cut_wa = cut_wb - gap
            cut_wb = cut_wa + t_w

            if cut_wb > w:
                cut_wb = w
                cut_wa = cut_wb - t_w
                W_OK =True
        if H_OK:
            break
        cut_ha = cut_hb - gap
        cut_hb = cut_ha + t_h

        if cut_hb > h:
            cut_hb = h
            cut_ha = cut_hb - t_h
            H_OK =True

    return img_cuts


if __name__ == '__main__':
    img = cv2.imread('F:/Projects/auto_Airplane/TS02/20191217_153659/894.png')
    imgs = img_cut(img)
