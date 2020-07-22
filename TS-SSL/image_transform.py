import random
import numpy as np
from scipy import misc

def im_crop(im):

    series = list([[1,2,3,4], [1,2,4,3], [1,3,2,4], [1,3,4,2], [1,4,2,3], [1,4,3,2],
                   [2,1,3,4], [2,1,4,3], [2,3,1,4], [2,3,4,1], [2,4,1,3], [2,4,3,1],
                   [3,1,2,4], [3,1,4,2], [3,2,1,4], [3,2,4,1], [3,4,1,2], [3,4,2,1],
                   [4,1,2,3], [4,1,3,2], [4,2,1,3], [4,2,3,1], [4,3,1,2], [4,3,2,1]])

    series_length = len(series)

    rand_idx = random.randint(0,series_length-1)
    rand_seri = series[rand_idx]
    #print(rand_seri)

    puzzle_label = np.zeros((series_length,1))
    puzzle_label[rand_idx,0] = 1
    #print(label)

    im_shape = im.shape
    h = im_shape[0]
    w = im_shape[1]
    #print(int(h/2))

    im_series = []
    im_p1 = im[0:int(h/2), 0:int(w/2)]
    im_series.append(im_p1)
    im_p2 = im[0:int(h/2), int(w/2):w]
    im_series.append(im_p2)
    im_p3 = im[int(h/2):h, 0:int(w/2)]
    im_series.append(im_p3)
    im_p4 = im[int(h/2):h, int(w/2):w]
    im_series.append(im_p4)

    im_vol = np.zeros((int(h), int(w)))
    rotated_label = np.zeros((4, 1, 4))
    
    rotated_patch, label = im_rotate(im_series[rand_seri[0]-1])
    rotated_label[0,:,:] = label
    im_vol[0:int(h/2), 0:int(w/2)] = rotated_patch
    
    rotated_patch, label = im_rotate(im_series[rand_seri[1]-1])
    rotated_label[1,:,:] = label
    im_vol[0:int(h/2), int(w/2):w] = rotated_patch
    
    rotated_patch, label = im_rotate(im_series[rand_seri[2]-1])
    rotated_label[2,:,:] = label
    im_vol[int(h/2):h, 0:int(w/2)] = rotated_patch
    
    rotated_patch, label = im_rotate(im_series[rand_seri[3]-1])
    rotated_label[3,:,:] = label
    im_vol[int(h/2):h, int(w/2):w] = rotated_patch

    return im_vol, puzzle_label, rotated_label

def im_rotate(im):

    im = np.array(im)

    rand_idx = random.randint(0,3)

    label = np.zeros((1,4))
    label[0, rand_idx] = 1

    
    if rand_idx==0:
        im_output = im
    if rand_idx==1:
        im_output = np.rot90(im,k=1,axes=(0,1))
    if rand_idx==2:
        im_output = np.rot90(im,k=2,axes=(0,1))
    if rand_idx==3:
        im_output = np.rot90(im,k=3,axes=(0,1))
        
    return im_output, label

def ori_cla_label(path):
    
    label = np.zeros((4,1))
            
    t1 = path.split('\\')
    t2 = t1[-1].split('-')
 
    if t2[0] == 'CNV':
        label[0] = 1
    if t2[0] == 'DME':
        label[1] = 1
    if t2[0] == 'DRUSEN':
        label[2] = 1
    if t2[0] == 'NORMAL':
        label[3] = 1

    return label
    


