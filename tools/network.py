# coding=utf-8
from tensorflow import keras
import efficientnet.tfkeras as ef
from tensorflow.keras import Input, Model, callbacks, applications, datasets, layers, losses, optimizers, activations
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Concatenate, Conv2D, UpSampling2D, BatchNormalization

import numpy as np
from PIL import Image
import cv2

from tools import utils


class East():

    def __init__(self):
        self.input_img = Input(name='input_img',
                               shape=(None, None, utils.num_channels),
                               dtype='float32')
        vgg16 = VGG16(input_tensor=self.input_img,
                      weights='imagenet',
                      include_top=False)
        if utils.locked_layers:
            # locked first two conv layers
            locked_layers = [vgg16.get_layer('block1_conv1'),
                             vgg16.get_layer('block1_conv2')]
            for layer in locked_layers:
                layer.trainable = False
        self.f = [vgg16.get_layer('block%d_pool' % i).output
                  for i in utils.feature_layers_range]
        self.f.insert(0, None)
        self.diff = utils.feature_layers_range[0] - utils.feature_layers_num

    def g(self, i):
        # i+diff in utils.feature_layers_range
        assert i + self.diff in utils.feature_layers_range, \
            ('i=%d+diff=%d not in ' % (i, self.diff)) + \
            str(utils.feature_layers_range)
        if i == utils.feature_layers_num:
            bn = BatchNormalization()(self.h(i))
            return Conv2D(32, 3, activation='relu', padding='same')(bn)
        else:
            return UpSampling2D((2, 2))(self.h(i))

    def h(self, i):
        # i+diff in utils.feature_layers_range
        assert i + self.diff in utils.feature_layers_range, \
            ('i=%d+diff=%d not in ' % (i, self.diff)) + \
            str(utils.feature_layers_range)
        if i == 1:
            return self.f[i]
        else:
            concat = Concatenate(axis=-1)([self.g(i - 1), self.f[i]])
            bn1 = BatchNormalization()(concat)
            conv_1 = Conv2D(128 // 2 ** (i - 2), 1,
                            activation='relu', padding='same',)(bn1)
            bn2 = BatchNormalization()(conv_1)
            conv_3 = Conv2D(128 // 2 ** (i - 2), 3,
                            activation='relu', padding='same',)(bn2)
            return conv_3

    def east_network(self):
        before_output = self.g(utils.feature_layers_num)
        inside_score = Conv2D(1, 1, padding='same', name='inside_score'
                              )(before_output)
        side_v_code = Conv2D(2, 1, padding='same', name='side_vertex_code'
                             )(before_output)
        side_v_coord = Conv2D(4, 1, padding='same', name='side_vertex_coord'
                              )(before_output)
        east_detect = Concatenate(axis=-1,
                                  name='east_detect')([inside_score,
                                                       side_v_code,
                                                       side_v_coord])
        return Model(inputs=self.input_img, outputs=east_detect)


class CRNN():
    
    def __init__(self, dropout=0.5, layer_nums=2, hidden_nums=256, num_classes=6796):
        self.dropout = dropout
        self.layer_nums = layer_nums
        self.hidden_nums =hidden_nums
        self.num_classes = num_classes
        
    def map_to_sequence(self, x):
        shape = x.get_shape().as_list()
        assert shape[-3]==1
        return keras.backend.squeeze(x, axis=-3)

    def blstm(self, x):
        x = layers.Lambda(lambda x: self.map_to_sequence(x))(x)
        for i in range(self.layer_nums):
            x = layers.Bidirectional(layers.LSTM(self.hidden_nums, return_sequences=True))(x)
        return x
    
    def crnn_network(self):
        print('Loading crnn backbone model.....')
        b7 = ef.EfficientNetB7(include_top=False, weights=None, input_shape=(32,None,3))
        print('Done!')
        y = keras.layers.MaxPool2D(pool_size=(4,1))(b7.get_layer(name='block4a_expand_activation').output)
        y = self.blstm(y)
        y = layers.Dropout(self.dropout)(y)
        y = layers.Dense(self.num_classes, activation='softmax', name='FC_1')(y)    
        crnn_model = Model(b7.inputs, y)
        print('Construct crnn_model Done!')
        
        return crnn_model

def predict_det(east_detect, img_path, pixel_threshold=0.9, quiet=False):
    img = image.load_img(img_path)
    # 以max_train_img_size为界等长宽比的缩放图片，并裁剪至32的整倍数, 返回w, h
    d_wight, d_height = utils.resize_image(img, utils.max_predict_img_size)
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    img = preprocess_input(img, mode='tf')
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)

    y = np.squeeze(y, axis=0)
    y[:, :, :3] = utils.sigmoid(y[:, :, :3])
    # np.greater_equal: >=
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    # 此处，数组是二维数组，返回满足条件的数组值的位置索引，因此会有两组索引数组来表示值的位置，
    # 返回的第一个array表示行坐标，第二个array表示纵坐标，两者一一对应
    activation_pixels = np.where(cond)
    # quad_after_nms：
    #            v1   v2
    #            v0   v3
    quad_scores, quad_after_nms = utils.nms(y, activation_pixels)
    return quad_scores, quad_after_nms, d_wight, d_height


def order_points(pts):
    # 一共4个坐标点
    rect = np.zeros((4, 2), dtype = "float32")

    # 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
    # 计算左上，右下
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 计算右上和左下
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    pts = np.array(pts).reshape(4,2)
#     print(pts)
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = image[y:y+h, x:x+w].copy()
#     print('croped.shape: ', croped.shape)
#     cv2.imwrite('./data/croped.jpg', croped)
    
    pts = pts - pts.min(axis=0)
    
    # 获取输入坐标点
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算输入的w和h值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 变换后对应坐标位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(croped, M, (maxWidth, maxHeight))

    # 返回变换后结果
    return warped


def predict_rec(crnn, img_path, quad_scores, quad_after_nms, d_wight, d_height, id2char):
    imgs = []
    img = cv2.imread(img_path)
    img = cv2.resize(img, (d_wight, d_height))
    # print(img.shape)
    coordinate = []
    for score, geo, s in zip(quad_scores, quad_after_nms, range(len(quad_scores))):
        if np.amin(score) > 0:
            coordinate.append(geo)
            if np.abs(geo[1][1] - geo[2][1]) / (np.abs(geo[1][1] - geo[0][1])) <= 0.2:
                top_left = (min(geo[:,0]), min(geo[:,1]))
    #         print(top_left)
                down_right = (max(geo[:,0]), max(geo[:,1]))
    #         print(down_right)
                imgs.append(img[int(top_left[1]):int(down_right[1]),
                                int(top_left[0]):int(down_right[0]), :])

            else:
                print('warped...')
                pst = np.array([int(geo[1][0]), int(geo[1][1]), int(geo[2][0]), int(geo[2][1]),
                                int(geo[3][0]), int(geo[3][1]), int(geo[0][0]), int(geo[0][1])])
                warped = four_point_transform(img, pst)
                imgs.append(warped)

    reg_imgs = []
    w_ = []
    for i, j  in enumerate(imgs):
        h,w,c = j.shape
        scale = h / 32
        ww = int(w * scale)
        reg_img = cv2.resize(j, (ww, 32))
        reg_imgs.append(reg_img)
        w_.append(ww)

    w = max(w_)
    mask = np.full((len(reg_imgs), 32, w, 3), fill_value=255)
    for i, j in enumerate(reg_imgs):
        mask[i, :, :j.shape[1], :] = j

    out = crnn.predict(mask)
    out_0 = keras.backend.ctc_decode(out, input_length=[out.shape[1]]*out.shape[0])[0]
    results = []
    for i, logit in enumerate(out_0[0].numpy()):
        
        result = [id2char[str(j+1)] for j in logit if j != -1]
        print(result)
        results.append(result)
        
    return results, coordinate