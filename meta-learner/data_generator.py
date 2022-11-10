""" Code for loading data. """
import numpy as np
import random
import tensorflow as tf

from tensorflow.python.platform import flags
import get_batch_task

FLAGS = flags.FLAGS

class DataGenerator(object):
    """
    Data Generator capable of generating batches of data.
    """
    def __init__(self, num_samples_per_class, batch_size, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class

        self.num_classes = config.get('num_classes', FLAGS.num_classes)
        self.img_size = config.get('img_size', (125, 30))
        self.dim_input = np.prod(self.img_size)*8
        self.dim_output = self.num_classes
        metatrain_folder = config.get('metatrain_folder', '/data/WN/data_icd_l1/train')
        if FLAGS.test_set:
            metaval_folder = config.get('metaval_folder', '/data/wn/data_icd_zhai/test') #test
        else:
            metaval_folder = config.get('metaval_folder', '/data/WN/data_icd_l1/train')

        self.metatrain_character_folders = metatrain_folder
        self.metaval_character_folders = metaval_folder

    def make_test_data_tensor(self, icd_count):
        folders = self.metaval_character_folders
        num_total_batches = 10
        print('Generating filenames')
        all_filenames = []
        for i in range(num_total_batches):
            all_filenames.extend(get_batch_task.taskgeneration(folders, self.num_samples_per_class, False))
        print('============================')
        print('all_filenames: ', len(all_filenames))
        temp = np.array(range(0,self.num_classes))
        labels = [val for val in temp for i in range(self.num_samples_per_class * icd_count)]
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print('Generating image processing ops')
        reader = tf.FixedLengthRecordReader(120000) 
        _, image_file = reader.read(filename_queue)
        data = tf.decode_raw(image_file, tf.float32)    
        data = tf.reshape(data, [125,30,8])
        
        print('============================')
        print('data: ', data)
        num_preprocess_threads = 1 # TODO - enable this to be set to >1
        min_queue_examples = 256
        examples_per_batch = self.num_classes * self.num_samples_per_class * icd_count 
        batch_image_size = self.batch_size  * examples_per_batch
        print('============================')
        print('examples_per_batch: ', examples_per_batch) 
        print('batch_image_size: ', batch_image_size)               
        print('self.num_classes: ', self.num_classes)
        print('self.num_samples_per_class: ', self.num_samples_per_class)
        print('self.batch_size: ', self.batch_size)
        images = tf.train.batch(
                [data],
                batch_size = batch_image_size,
                num_threads = num_preprocess_threads,
                capacity = min_queue_examples + 3 * batch_image_size,
                )
        all_image_batches, all_label_batches = [], []
        print('Manipulating image data to be right shape')
        for i in range(self.batch_size):
            image_batch = images[i*examples_per_batch:(i+1)*examples_per_batch]
            label_batch = tf.convert_to_tensor(labels)
            new_list, new_label_list = [], []
            
            random_bias = random.randint(0, self.num_samples_per_class * icd_count)
            for k in range(self.num_samples_per_class * icd_count):
                class_idxs = tf.range(0, self.num_classes)
                class_idxs = tf.random.shuffle(class_idxs)

                true_idxs = class_idxs*self.num_samples_per_class*icd_count + (k + random_bias) % (self.num_samples_per_class*icd_count)
                new_list.append(tf.gather(image_batch,true_idxs))
                
                new_label_list.append(tf.gather(label_batch, true_idxs))
            new_list = tf.concat(new_list, 0)
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
            
        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        print('==============')
        print('all_image_batches: ', all_image_batches)
        print('all_label_batches: ', all_label_batches)
        print('==============')
        return all_image_batches, all_label_batches

    def make_data_tensor(self, train=True):
        if train:
            folders = self.metatrain_character_folders
            num_total_batches = 300 
        else:
            folders = self.metaval_character_folders
            num_total_batches = 15 
            
        print('Generating filenames')
        all_filenames = []
        for i in range(num_total_batches):
            all_filenames.extend(get_batch_task.taskgeneration(folders, self.num_samples_per_class, train))
        print('============================')
        print('all_filenames: ', len(all_filenames))
        temp = np.array(range(0,self.num_classes))
        labels = [val for val in temp for i in range(self.num_samples_per_class)]
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print('Generating image processing ops')
        reader = tf.FixedLengthRecordReader(120000)
        _, image_file = reader.read(filename_queue)
        data = tf.decode_raw(image_file, tf.float32)    
        data = tf.reshape(data, [125,30,8])
        print('============================')
        print('data: ', data)
        num_preprocess_threads = 1
        min_queue_examples = 256
        examples_per_batch = self.num_classes * self.num_samples_per_class
        batch_image_size = self.batch_size  * examples_per_batch
        print('============================')
        print('batch_image_size: ', batch_image_size)               
        print('self.num_classes: ', self.num_classes)
        print('self.num_samples_per_class: ', self.num_samples_per_class)
        print('self.batch_size: ', self.batch_size)
        images = tf.train.batch(
                [data],
                batch_size = batch_image_size,
                num_threads = num_preprocess_threads,
                capacity = min_queue_examples + 3 * batch_image_size,
                )
        all_image_batches, all_label_batches = [], []
        print('Manipulating image data to be right shape')
        for i in range(self.batch_size):
            image_batch = images[i*examples_per_batch:(i+1)*examples_per_batch]
            label_batch = tf.convert_to_tensor(labels)
            new_list, new_label_list = [], []
            for k in range(self.num_samples_per_class):
                class_idxs = tf.range(0, self.num_classes)
                class_idxs = tf.random.shuffle(class_idxs)

                true_idxs = class_idxs*self.num_samples_per_class + k
                new_list.append(tf.gather(image_batch,true_idxs))
                
                new_label_list.append(tf.gather(label_batch, true_idxs))
            new_list = tf.concat(new_list, 0)
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
       
        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        print('==============')
        print('all_image_batches: ', all_image_batches)
        print('all_label_batches: ', all_label_batches)
        print('==============')
        return all_image_batches, all_label_batches

