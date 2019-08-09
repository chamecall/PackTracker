import cv2
import os
import numpy as np
import math
import json
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose
)
class Picture:
    @staticmethod
    def get_img_name_and_json(first_word, number, format_img = 'png', format_label='json'):
        file_name = f'{first_word}{number}).{format_img}'
        json_name = f'{first_word}{number}).{format_label}'
        return file_name, json_name

    @staticmethod
    def upload(catalog, img_name):
        img = None
        img_dir = os.path.join(catalog, img_name)
        try:
            if os.path.exists(img_dir):
                img = cv2.imread(img_dir)
        except:
            print(f"Upload Error. File: {img_dir}")
        return img

    @staticmethod
    def set_augment(aug, img):
        img = aug(image=img)['image']
        return img

    @staticmethod
    def load(catalog, result_img_name, image):
        res_dir = os.path.join(catalog, result_img_name)
        cv2.imwrite(res_dir, image)

    @staticmethod
    def augment_flips_color(p=.5):
        return Compose([
            CLAHE(p=.2),
            # RandomRotate90(p=.5),
            # Transpose(),
            ShiftScaleRotate(shift_limit=0.0325, scale_limit=0.20, rotate_limit=45, p=.6),
            # Blur(blur_limit=3),
            # OpticalDistortion(),
            # GridDistortion(),
            HueSaturationValue(sat_shift_limit=10,hue_shift_limit=10, p=.3)
        ], p=p)

    @staticmethod
    def img_generation(catalog='D:\public\Screenshots', res_catalog='Result',
                       first_word_name='Screenshot (', img_count=208, count_random=3,
                       curr_ind=159, res_ind=393):
        res_catalog = os.path.join(catalog, res_catalog)
        while (curr_ind <= img_count):
            img_name, json_name = Picture.get_img_name_and_json(first_word_name, curr_ind)
            img = Picture.upload(catalog, img_name)
            if (img is None):
                curr_ind += 1
                continue
            h, w = img.shape[0], img.shape[1]
            for i in range(count_random):
                aug = Picture.get_aug()

                #   change img

                # change channel
                img = Picture.set_better_channel(img, 2.5)
                # change position
                img = Picture.shift_scale_rotate(img, aug['angle'], aug['scale'], aug['dx'], aug['dy'])

                # points = Picture.get_points(catalog, json_name)
                # new_points = []
                # for i in points:
                #     new_points.append(
                #         Picture.keypoint_shift_scale_rotate(i, aug['angle'], aug['scale'], aug['dx'], aug['dy'], h, w))

                res_ind += 1
                res_img_name, res_json_name = Picture.get_img_name_and_json(first_word_name, res_ind)
                # Picture.set_points_in_JSON(new_points, catalog, json_name, res_catalog, res_json_name, res_img_name)
                Picture.load(res_catalog, res_img_name, img)
            curr_ind += 1
            print(curr_ind)

    @staticmethod
    def keypoint_shift_scale_rotate(keypoint, angle, scale, dx, dy, rows, cols):
        height, width = rows, cols
        center = (width / 2, height / 2)
        label = keypoint[0]
        points = []
        x = keypoint[1][0]
        y = keypoint[1][1]
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        matrix[0, 2] += dx * width
        matrix[1, 2] += dy * height
        x, y = cv2.transform(np.array([[x, y]]), matrix).squeeze()
        a = [x[0], x[1]]
        b = [y[0], y[1]]
        points.append((label, (x,y)))
        return (label, [a,b])

    @staticmethod
    def shift_scale_rotate(img, angle, scale, dx, dy, interpolation=cv2.INTER_LINEAR,
                           border_mode=cv2.BORDER_REFLECT_101,
                           value=None):
        height, width = img.shape[:2]
        center = (width / 2, height / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        matrix[0, 2] += dx * width
        matrix[1, 2] += dy * height
        img = cv2.warpAffine(img, matrix, (width, height), flags=interpolation, borderMode=border_mode,
                             borderValue=value)
        return img

    @staticmethod
    def get_aug():
        aug = ShiftScaleRotate(shift_limit=0.0325, scale_limit=0.20, rotate_limit=15, p=1)
        return aug.get_params()

    # :TODO
    @staticmethod
    def get_points(catalog, name_json):
        points=[]
        name_json = os.path.join(catalog, name_json)
        with open(name_json) as js_file:
            data = json.load(js_file)
            for p in data['shapes']:
                points.append((p['label'], p['points']))
        return points

    @staticmethod
    def set_points_in_JSON(list_points, catalog_orig, name_orig_json,catalog_result, name_result_json, name_path):
        name_orig_json= os.path.join(catalog_orig, name_orig_json)
        name_result_json = os.path.join(catalog_result, name_result_json)
        ind = 0
        with open(name_orig_json) as orig_js:
            data = json.load(orig_js)
        data['imagePath'] = name_path
        data['imageData'] = None
        for point in list_points:
            data['shapes'][ind]['label'] = point[0]
            x = point[1]
            data['shapes'][ind]['points'] = x
            ind +=1
        with open( name_result_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


    @staticmethod
    def set_better_channel(img, clpLim = 3.0, tileGridSize = (8, 8)):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clpLim, tileGridSize=tileGridSize)
        cl = clahe.apply(l)

        limg = cv2.merge((cl, a, b))

        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final

# edges = cv2.Canny(img,100,200)
# plt.subplot(121), plt.imshow(img, cmap='gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(edges, cmap='gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

# catalog = 'D:\public\Screenshots'
# res_catalog = os.path.join(catalog, 'Result')
#
# first_word_name = 'Screenshot ('
#
# img_count = 139
# count_random = 5
#
# curr_ind = 4
# res_ind = 140


# Picture.img_generation('Images\Class','Result')
