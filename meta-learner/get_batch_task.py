# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:49:49 2020

@author: Coolnerdn
"""

import os
import random
from utils import get_images

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

def taskgeneration(meta_folder, num_samples_per_class, is_train) :
    all_filenames = []
    if is_train: # train
        folders = [os.path.join(meta_folder, pNo) \
                    for pNo in os.listdir(meta_folder) \
                    if os.path.isdir(os.path.join(meta_folder, pNo)) \
                    ]
        # per-condition tasks
        for i in range(len(folders)):
            # filefolders = [os.path.join(folders[dno], cNo) for cNo in os.listdir(folders[dno])]
            filefolders = [os.path.join(folders[i], cNo) for cNo in os.listdir(folders[i])]
            random.shuffle(filefolders)
            labels_and_images = get_images(filefolders, range(6), nb_samples=num_samples_per_class, shuffle=False) # 60
            filenames = [li[1] for li in labels_and_images]
            all_filenames.extend(filenames)
            
        # multi-conditioned tasks
        for _ in range(len(folders)):
            for cno in range(1,7):
                pno = random.randint(0, len(folders)-1)
                filefolders = [os.path.join(folders[pno], str(cno))]
                labels_and_images = get_images(filefolders, range(6), nb_samples=num_samples_per_class, shuffle=False)
                filenames = [li[1] for li in labels_and_images]
                all_filenames.extend(filenames)
    
    else: # val & test
        folders = [os.path.join(meta_folder, pNo) \
                    for pNo in os.listdir(meta_folder) \
                    if os.path.isdir(os.path.join(meta_folder, pNo)) \
                    ]
        random_bias = random.randint(0, FLAGS.all_icd_count)
        for cno in range(1,7):
            for samples in range(num_samples_per_class):
                filenames = [os.path.join(folders[(dno+random_bias)%FLAGS.all_icd_count], str(cno), str(samples)+".bin") for dno in range(FLAGS.icd_count)]
                all_filenames.extend(filenames)
        
    return all_filenames

