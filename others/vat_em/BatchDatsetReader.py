import numpy as np
import scipy.io
from scipy import misc
import os
import random
import h5py
import image_transform as it

class BatchDatset:

    def __init__(self, path):
        print("Initializing Batch Dataset Reader...")
        self.path_list = [];
        self.batch_offset = 0
        self.epochs_completed = 0
        self.path = path
        self._all_path()

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > len(self.path_list):
            self.epochs_completed += 1
            #print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            random.shuffle(self.path_list)
            start = 0
            self.batch_offset = batch_size
        end = self.batch_offset
        path_temp = self.path_list[start:end]

        images = np.arange(batch_size*512*512*1).reshape(batch_size,512,512,1)
        annotations = np.arange(batch_size*2*1).reshape(batch_size,2,1)
        
        k = -1
        for subpath in path_temp:
            #print(subpath)
            im = np.array(misc.imread(subpath))
            train_img = misc.imresize(im, [512, 512], interp='nearest')
            train_label = it.ori_cla_label(subpath)

            k = k+1
            images[k,:,:,0] = train_img
            annotations[k,:,:] = train_label
        #print(annotations)
            
        return images, annotations


    def _all_path(self):
        for maindir, subdir, file_name_list in os.walk(self.path):
            for filename in file_name_list:
                apath = os.path.join(maindir, filename)
                self.path_list.append(apath)
        random.shuffle(self.path_list)
