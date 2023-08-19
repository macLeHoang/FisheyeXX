# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import random

def flip(img):
  return img[:, :, ::-1].copy()  

# @numba.jit(nopython=True, nogil=True)
def transform_preds_with_trans(coords, trans):
    # target_coords = np.concatenate(
    #   [coords, np.ones((coords.shape[0], 1), np.float32)], axis=1)
    target_coords = np.ones((coords.shape[0], 3), np.float32)
    target_coords[:, :2] = coords
    target_coords = np.dot(trans, target_coords.transpose()).transpose()
    return target_coords[:, :2]


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)
    # print(center, scale, rot, output_size)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img

# @numba.jit(nopython=True, nogil=True)
def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)


# @numba.jit(nopython=True, nogil=True)
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    # y, x = np.arange(-m, m + 1).reshape(-1, 1), np.arange(-n, n + 1).reshape(1, -1)
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

# @numba.jit(nopython=True, nogil=True)
def draw_umich_gaussian(heatmap, center, radius, k=1):
  # import pdb; pdb.set_trace()
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]
    
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)
  # import pdb; pdb.set_trace()
  masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap


def draw_umich_gaussian_1d(heatmap, center, radius, k=1):
    # import pdb; pdb.set_trace()
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    # import pdb; pdb.set_trace()
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
  dim = value.shape[0]
  reg = np.ones((dim, diameter*2+1, diameter*2+1), dtype=np.float32) * value
  if is_offset and dim == 2:
    delta = np.arange(diameter*2+1) - radius
    reg[0] = reg[0] - delta.reshape(1, -1)
    reg[1] = reg[1] - delta.reshape(-1, 1)
  
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]
    
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
  masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom,
                             radius - left:radius + right]
  masked_reg = reg[:, radius - top:radius + bottom,
                      radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    idx = (masked_gaussian >= masked_heatmap).reshape(
      1, masked_gaussian.shape[0], masked_gaussian.shape[1])
    masked_regmap = (1-idx) * masked_regmap + idx * masked_reg
  regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
  return regmap


def draw_msra_gaussian(heatmap, center, sigma):
  tmp_size = sigma * 3
  mu_x = int(center[0] + 0.5)
  mu_y = int(center[1] + 0.5)
  w, h = heatmap.shape[0], heatmap.shape[1]
  ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
  br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
  if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
    return heatmap
  size = 2 * tmp_size + 1
  x = np.arange(0, size, 1, np.float32)
  y = x[:, np.newaxis]
  x0 = y0 = size // 2
  g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
  g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
  g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
  img_x = max(0, ul[0]), min(br[0], h)
  img_y = max(0, ul[1]), min(br[1], w)
  heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
  return heatmap

def grayscale(image):
  return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def lighting_(data_rng, image, alphastd, eigval, eigvec):
  alpha = data_rng.normal(scale=alphastd, size=(3, ))
  image += np.dot(eigvec, eigval * alpha)

def blend_(alpha, image1, image2):
  image1 *= alpha
  image2 *= (1 - alpha)
  image1 += image2

def saturation_(data_rng, image, gs, gs_mean, var):
  alpha = 1. + data_rng.uniform(low=-var, high=var)
  blend_(alpha, image, gs[:, :, None])

def brightness_(data_rng, image, gs, gs_mean, var):
  alpha = 1. + data_rng.uniform(low=-var, high=var)
  image *= alpha

def contrast_(data_rng, image, gs, gs_mean, var):
  alpha = 1. + data_rng.uniform(low=-var, high=var)
  blend_(alpha, image, gs_mean)

def color_aug(data_rng, image, eig_val, eig_vec):
  functions = [brightness_, contrast_, saturation_]
  random.shuffle(functions)

  gs = grayscale(image)
  gs_mean = gs.mean()
  for f in functions:
      f(data_rng, image, gs, gs_mean, 0.4)
  lighting_(data_rng, image, 0.1, eig_val, eig_vec)


def get_intersection(points, w, h, oriented_angle=None, rot=None, pre_trained=False):
  xmin, y_xmin = points[0]
  x_ymin, ymin = points[1]
  xmax, y_xmax = points[2]
  x_ymax, ymax = points[3]

  xmin_news = []
  xmax_news = []
  ymin_news = []
  ymax_news = []

  # alpha * x_ + (1-alpha)* x_{max | min}  = 0
  # xmin
  if xmin < 0:
      if x_ymin < 0 and x_ymax < 0:
        alpha1 = xmax / (xmax - x_ymin)
        new_y1 = alpha1 * ymin + (1-alpha1) * y_xmax
        alpha2 = xmax / (xmax - x_ymax)
        new_y2 = alpha2 * ymax + (1-alpha2) * y_xmax

      elif x_ymin < 0:
        alpha1 = xmax / (xmax - x_ymin)
        new_y1 = alpha1 * ymin + (1-alpha1) * y_xmax
        alpha2 = xmin / (xmin - x_ymax)
        new_y2 = alpha2 * ymax + (1-alpha2) * y_xmin

      elif x_ymax < 0:
        alpha1 = xmin / (xmin - x_ymin)
        new_y1 = alpha1 * ymin + (1-alpha1) * y_xmin
        alpha2 = xmax / (xmax - x_ymax)
        new_y2 = alpha2 * ymax + (1-alpha2) * y_xmax

      else:
        alpha1 = xmin / (xmin - x_ymin)
        new_y1 = alpha1 * ymin + (1-alpha1) * y_xmin
        alpha2 = xmin / (xmin - x_ymax)
        new_y2 = alpha2 * ymax + (1-alpha2) * y_xmin

      if -1 < new_y1 < h and -1 < new_y2 < h:
        if new_y1 > new_y2:
          xmin_news.append((0, new_y1))
          xmin_news.append((0, new_y2))
        else:
          xmin_news.append((0, new_y2))
          xmin_news.append((0, new_y1))
      elif -1 < new_y1 < h:
        xmin_news.append((0, new_y1))
      elif -1 < new_y2 < h:
        xmin_news.append((0, new_y2)) 
  else:
    if -1 < y_xmin < h and -1 < xmin < w:
      xmin_news.append((xmin, y_xmin))
  
  # xmax
  if xmax > w-1:
      if x_ymin > w-1 and x_ymax > w-1:
        alpha1 = (xmin - (w-1)) / (xmin - x_ymin)
        new_y1 = alpha1 * ymin + (1-alpha1) * y_xmin
        alpha2 = (xmin - (w-1)) / (xmin - x_ymax)
        new_y2 = alpha2 * ymax + (1-alpha2) * y_xmin

      elif x_ymin > w-1:
        alpha1 = (xmin - (w-1)) / (xmin - x_ymin)
        new_y1 = alpha1 * ymin + (1-alpha1) * y_xmin
        alpha2 = (xmax - (w-1)) / (xmax - x_ymax)
        new_y2 = alpha2 * ymax + (1-alpha2) * y_xmax

      elif x_ymax > w-1:
        alpha1 = (xmax - (w-1)) / (xmax - x_ymin)
        new_y1 = alpha1 * ymin + (1-alpha1) * y_xmax
        alpha2 = (xmin - (w-1)) / (xmin - x_ymax)
        new_y2 = alpha2 * ymax + (1-alpha2) * y_xmin

      else:
        alpha1 = (xmax - (w-1)) / (xmax - x_ymin)
        new_y1 = alpha1 * ymin + (1-alpha1) * y_xmax
        alpha2 = (xmax - (w-1)) / (xmax - x_ymax)
        new_y2 = alpha2 * ymax + (1-alpha2) * y_xmax

      if -1 < new_y1 < h and -1 < new_y2 < h:
        if new_y1 < new_y2:
          xmax_news.append((w-1, new_y1))
          xmax_news.append((w-1, new_y2))
        else:
          xmax_news.append((w-1, new_y2))
          xmax_news.append((w-1, new_y1))
      elif -1 < new_y1 < h:
        xmax_news.append((w-1, new_y1))
      elif -1 < new_y2 < h:
        xmax_news.append((w-1, new_y2)) 
  else:
    if -1 < y_xmax < h and -1 < xmax < w:
      xmax_news.append((xmax, y_xmax))

  # ymin
  if ymin < 0:
      if y_xmin < 0 and y_xmax < 0:
        alpha1 = ymax / (ymax - y_xmin)
        new_x1 = alpha1 * xmin + (1-alpha1) * x_ymax
        alpha2 = ymax / (ymax - y_xmax)
        new_x2 = alpha2 * xmax + (1-alpha2) * x_ymax
      
      elif y_xmin < 0:
        alpha1 = ymax / (ymax - y_xmin)
        new_x1 = alpha1 * xmin + (1-alpha1) * x_ymax
        alpha2 = ymin / (ymin - y_xmax)
        new_x2 = alpha2 * xmax + (1-alpha2) * x_ymin
      
      elif y_xmax < 0:
        alpha1 = ymin / (ymin - y_xmin)
        new_x1 = alpha1 * xmin + (1-alpha1) * x_ymin
        alpha2 = ymax / (ymax - y_xmax)
        new_x2 = alpha2 * xmax + (1-alpha2) * x_ymax

      else:
        alpha1 = ymin / (ymin - y_xmin)
        new_x1 = alpha1 * xmin + (1-alpha1) * x_ymin
        alpha2 = ymin / (ymin - y_xmax)
        new_x2 = alpha2 * xmax + (1-alpha2) * x_ymin

      if -1 < new_x1 < w and -1 < new_x2 < w:
        if new_x1 < new_x2: 
          ymin_news.append((new_x1, 0))
          ymin_news.append((new_x2, 0))
        else:
          ymin_news.append((new_x2, 0))
          ymin_news.append((new_x1, 0))
      elif -1 < new_x1 < w:
        ymin_news.append((new_x1, 0))
      elif -1 < new_x2 < w:
        ymin_news.append((new_x2, 0))
  else:
    if -1 < x_ymin < w and -1 < ymin < h:
      ymin_news.append((x_ymin, ymin))

  # ymax
  if ymax > h-1:
      if y_xmin > h-1 and y_xmax > h-1:
        alpha1 = (ymin - (h-1)) / (ymin - y_xmin)
        new_x1 = alpha1 * xmin + (1-alpha1) * x_ymin
        alpha2 = (ymin - (h-1)) / (ymin - y_xmax)
        new_x2 = alpha2 * xmax + (1-alpha2) * x_ymin

      elif y_xmin > h-1:
        alpha1 = (ymin - (h-1)) / (ymin - y_xmin)
        new_x1 = alpha1 * xmin + (1-alpha1) * x_ymin
        alpha2 = (ymax - (h-1)) / (ymax - y_xmax)
        new_x2 = alpha2 * xmax + (1-alpha2) * x_ymax

      elif y_xmax > h-1:
        alpha1 = (ymax - (h-1)) / (ymax - y_xmin)
        new_x1 = alpha1 * xmin + (1- alpha1) * x_ymax
        alpha2 = (ymin - (h-1)) / (ymin - y_xmax)
        new_x2 = alpha2 * xmax + (1-alpha2) * x_ymin
      
      else:
        alpha1 = (ymax - (h-1)) / (ymax - y_xmin)
        new_x1 = alpha1 * xmin + (1- alpha1) * x_ymax
        alpha2 = (ymax - (h-1)) / (ymax - y_xmax)
        new_x2 = alpha2 * xmax + (1-alpha2) * x_ymax
        
      if -1 < new_x1 < w and -1 < new_x2 < w:
        if new_x1 < new_x2:
          ymax_news.append((new_x1, h-1))
          ymax_news.append((new_x2, h-1))
        else:
          ymax_news.append((new_x2, h-1))
          ymax_news.append((new_x1, h-1))
      elif -1 < new_x1 < w:
        ymax_news.append((new_x1, h-1))
      elif -1 < new_x2 < w:
        ymax_news.append((new_x2, h-1))
  else:
    if -1 < x_ymax < w and -1 < ymax < h:
      ymax_news.append((x_ymax, ymax))

  total_points = len(xmin_news) + len(xmax_news) + len(ymin_news) + len(ymax_news)
  if total_points == 0:
    return [0,0,0,0,0]

  points = xmin_news + xmax_news + ymin_news + ymax_news
  points = np.vstack(points)

  
  if total_points == 3 or total_points == 8:
    xmin, xmax = points[:, 0].min(), points[:, 0].max()
    ymin, ymax = points[:, 1].min(), points[:, 1].max()

    return [xmin, ymax, xmax, ymax, 0]
  else:
    xmin_news = np.array(xmin_news) 
    xmax_news = np.array(xmax_news)
    ymin_news = np.array(ymin_news) 
    ymax_news = np.array(ymax_news)
    
    if xmin_news.shape[0] == 0 or ymin_news.shape[0]==0:
      w1=0
    else:
      w1 = ((xmin_news[0] - ymin_news[0])**2).sum() **0.5
    if xmax_news.shape[0]==0 or ymax_news.shape[0]==0:
      w2 = 0
    else:
      w2 = ((xmax_news[-1] - ymax_news[-1])**2).sum() **0.5
    if w1==0 or w2==0:
      w = max(w1, w2)*0.79
    else: 
      w = (w1+w2)*0.7 if min(w1,w2)/max(w1,w2) < 0.89 else min(w1,w2)

    if xmin_news.shape[0]==0 or ymax_news.shape[0]==0:
      h1 = 0
    else:
      h1 = ((xmin_news[-1] - ymax_news[0])**2).sum() **0.5
    if xmax_news.shape[0]==0 or ymin_news.shape[0]==0:
      h2 = 0
    else:
      h2 = ((xmax_news[0] - ymin_news[-1])**2).sum() **0.5
    if h1==0 or h2==0:
      h = max(h1,h2)*0.79
    else:
      h = (h1+h2)*0.7 if min(h1,h2)/max(h1,h2) < 0.89 else min(h1, h2)

    oriented_angle -= rot
    oriented_angle = oriented_angle % 180 - 180 \
                if oriented_angle % 180 > 90 \
                else oriented_angle % 180

    
    if not pre_trained:
      if h > w:
        oriented_angle -= 90
      elif w > h:
        w, h = h, w
    else:
      # switch between w and h ??
      if 0 < oriented_angle <= 90:
        w, h = h, w 

    if oriented_angle == 90:
      oriented_angle = -90.0

    xc, yc = points.sum(axis = 0) / total_points

    return [xc-w/2, yc-h/2, xc+w/2, yc+h/2, oriented_angle]