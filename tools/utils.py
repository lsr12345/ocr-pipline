import os
import numpy as np
from PIL import Image, ImageDraw

max_predict_img_size = 512  # 2400
epsilon = 1e-4

num_channels = 3
feature_layers_range = range(5, 1, -1)
# feature_layers_range = range(3, 0, -1)
feature_layers_num = len(feature_layers_range)
# pixel_size = 4
pixel_size = 2 ** 2
locked_layers = False

side_vertex_pixel_threshold = 0.9
trunc_threshold = 0.1

def point_inside_of_quad(self, px, py, quad_xy_list, p_min, p_max):
    if (p_min[0] <= px <= p_max[0]) and (p_min[1] <= py <= p_max[1]):
        xy_list = np.zeros((4, 2))
        xy_list[:3, :] = quad_xy_list[1:4, :] - quad_xy_list[:3, :]
        xy_list[3] = quad_xy_list[0, :] - quad_xy_list[3, :]
        yx_list = np.zeros((4, 2))
        yx_list[:, :] = quad_xy_list[:, -1:-3:-1]
        a = xy_list * ([py, px] - yx_list)
        b = a[:, 0] - a[:, 1]
        if np.amin(b) >= 0 or np.amax(b) <= 0:
            return True
        else:
            return False
    else:
        return False

def resize_image(im, max_img_size):

    im_width = np.minimum(im.width, max_img_size)
    if im_width == max_img_size < im.width:
        im_height = int((im_width / im.width) * im.height)
    else:
        im_height = im.height
    o_height = np.minimum(im_height, max_img_size)
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:
        o_width = im_width
    d_wight = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_wight, d_height

def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1.0 / (1.0 + np.exp(-x))


def cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array, img_path, s):
    geo /= [scale_ratio_w, scale_ratio_h]
    p_min = np.amin(geo, axis=0)
    p_max = np.amax(geo, axis=0)
    min_xy = p_min.astype(int)
    max_xy = p_max.astype(int) + 2
    sub_im_arr = im_array[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0], :].copy()
    for m in range(min_xy[1], max_xy[1]):
        for n in range(min_xy[0], max_xy[0]):
            if not point_inside_of_quad(n, m, geo, p_min, p_max):
                sub_im_arr[m - min_xy[1], n - min_xy[0], :] = 255
    sub_im = image.array_to_img(sub_im_arr, scale=False)
    sub_im.save(img_path + '_subim%d.jpg' % s)

    
def should_merge(region, i, j):
    neighbor = {(i, j - 1)}
    return not region.isdisjoint(neighbor)


def region_neighbor(region_set):
    region_pixels = np.array(list(region_set))
    j_min = np.amin(region_pixels, axis=0)[1] - 1
    j_max = np.amax(region_pixels, axis=0)[1] + 1
    i_m = np.amin(region_pixels, axis=0)[0] + 1
    region_pixels[:, 0] += 1
    neighbor = {(region_pixels[n, 0], region_pixels[n, 1]) for n in
                range(len(region_pixels))}
    neighbor.add((i_m, j_min))
    neighbor.add((i_m, j_max))
    return neighbor

# m: 元组region_list的索引  S: 元组region_list的索引集合
# 返回 经过合并后的区域点索引 所组成的元组集合
def region_group(region_list):
    S = [i for i in range(len(region_list))]
    D = []
    while len(S) > 0:
        m = S.pop(0)
        if len(S) == 0:
            # S has only one element, put it to D
            D.append([m])
        else:
            D.append(rec_region_merge(region_list, m, S))
    return D

# 除去region_list的索引集合S中相互包含的元组，返回剩余元组列表
# m n 分别表示一组相邻的区域点集合
# 功能：与 m区域点集有交集的其他集合全部并入m中，并在S中remove这些被合并的集合
def rec_region_merge(region_list, m, S):
    rows = [m]
    tmp = []
    for n in S:
        # set.isdisjoint(set) 判断两个集合是否包含相同的元素，如果不包含返回 True，包含返回 False
        if not region_neighbor(region_list[m]).isdisjoint(region_list[n]) or \
                not region_neighbor(region_list[n]).isdisjoint(region_list[m]):
            # 第m与n相交
            tmp.append(n)
    for d in tmp:
        S.remove(d)
    for e in tmp:
        rows.extend(rec_region_merge(region_list, e, S))
    return rows


def nms(predict, activation_pixels, threshold=side_vertex_pixel_threshold):
    region_list = []
    # a = [1,2,3] b = [4,5,6] -> c = zip(a,b) = [(1,4),(2,5),(3,6)]
    # activation_pixels[0]：行坐标的array, activation_pixels[1]: 列坐标的array
    for i, j in zip(activation_pixels[0], activation_pixels[1]):
        merge = False
        for k in range(len(region_list)):
            # 如果 region_list[k]包含（i, j-1）点， merge邻近点进一个元组
            if should_merge(region_list[k], i, j):
                region_list[k].add((i, j))
                merge = True
                # Fixme 重叠文本区域处理，存在和多个区域邻接的pixels，先都merge试试
                # break
        if not merge:
            region_list.append({(i, j)})
    # 返回 经过合并后的区域点索引 所组成的元组集合
    D = region_group(region_list)
    # quad_list：
    #            v1   v2
    #            v0   v3
    quad_list = np.zeros((len(D), 4, 2))
    score_list = np.zeros((len(D), 4))
    for group, g_th in zip(D, range(len(D))):
        total_score = np.zeros((4, 2))
        for row in group:
            for ij in region_list[row]:
                score = predict[ij[0], ij[1], 1]
                if score >= threshold:
                    ith_score = predict[ij[0], ij[1], 2:3]
                    if not (trunc_threshold <= ith_score < 1 -
                            trunc_threshold):
                        # np.around 返回四舍五入后的值  ith: 0 or 1
                        ith = int(np.around(ith_score))
                        total_score[ith * 2:(ith + 1) * 2] += score
                        px = (ij[1] + 0.5) * pixel_size
                        py = (ij[0] + 0.5) * pixel_size
                        # 某个边界点预测的2个顶点坐标
                        p_v = [px, py] + np.reshape(predict[ij[0], ij[1], 3:7],
                                              (2, 2))
                        # 预测的2个顶点坐标*分数权重
                        quad_list[g_th, ith * 2:(ith + 1) * 2] += score * p_v
        # 边界点的总得分
        score_list[g_th] = total_score[:, 0]
        # 利用边界点总得分对得到的顶点坐标进行归一化
        quad_list[g_th] /= (total_score + epsilon)
    return score_list, quad_list
