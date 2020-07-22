import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread,imresize
import os
import os.path

smooth = 1.

def dice_coef(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def jaccard_coef(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth - intersection)

def normalize(img, s=0.1):
    """Normalize the image range for visualization"""
    z = img / np.std(img)
    return np.uint8(np.clip((z - z.mean()) / max(z.std(), 1e-4) * s + 0.5, 0, 1) * 255)

def predict (output):
    inference = 1 / (1 +  np.e**(-output))  #sigmoid用于2分类，softmax用于多分类
    text = '%.2f%%' % (inference * 100)
    return text

def get_cam_up(label_w, fmaps,height = 224, width = 224, num_fmaps = 1472):
    #fmaps_resized = tf.image.resize_bilinear( fmaps, [height, width] )
    fmaps_resized = fmaps
    fmaps_resized = fmaps_resized.reshape( [-1, height*width, num_fmaps])
    classmap = np.matmul( fmaps_resized, label_w )
    classmap = np.reshape( classmap, [-1, height, width] )
    return np.squeeze(classmap,axis=(0,))

def bounding_column(classmap,height,wdith,threshold):
    bounding = classmap
    col = []
    for i in range(wdith):
        for j in range(height):
            if bounding[j][i] >= threshold:
                col.append(i)
                break
    bounding[:,:] = 0
    for i in col:
        bounding[:,i] = 1
    return bounding

def getFiles(dir, suffix):
    res = []
    for root, directory, files in os.walk(dir):
        for filename in files:
            name, suf = os.path.splitext(filename)
            if suf == suffix:
                res.append(os.path.join(root, filename))
    return res

def getProjection(img):
    '''
    Proj = []
    for i in range(img.shape[1]):
        Proj.append(img[:,i].sum())
    return Proj
    '''
    return(np.sum(img,axis=0))



