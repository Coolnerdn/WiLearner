""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import flags
from tensorflow.keras import layers
from utils import xent, normalize

FLAGS = flags.FLAGS

class MAML:
    def __init__(self, dim_input=1, dim_output=1, test_num_updates=5):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.compat.v1.placeholder_with_default(FLAGS.meta_lr, ())
        self.classification = False
        self.test_num_updates = test_num_updates # test:30, train:1
        self.loss_func = xent
        self.classification = True
        if FLAGS.conv:
            self.dim_hidden = FLAGS.num_filters
            self.forward = self.forward_conv
            self.construct_weights = self.construct_conv_weights
        elif FLAGS.resnet:
            self.forward = self.forward_resnet
            self.construct_weights = self.construct_resnet_weights
        else:
            self.dim_hidden = [256, 128, 64, 64]
            self.forward=self.forward_fc
            self.construct_weights = self.construct_fc_weights
        self.channels = 8
        self.img_size = (125, 30) #int(np.sqrt(self.dim_input/self.channels))

    def construct_model(self, input_tensors=None, prefix='metatrain_'):
        # a: training data for inner gradient, b: test data for meta gradient
        if input_tensors is None:
            self.inputa = tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)
        else:
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']

        with tf.compat.v1.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()

            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            num_updates = max(self.test_num_updates, FLAGS.num_updates)
            outputbs = [[]]*num_updates
            lossesb = [[]]*num_updates
            accuraciesb = [[]]*num_updates

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb = [], []

                if self.classification:
                    task_accuraciesb = []
                task_outputa = self.forward(inputa, weights, reuse=reuse)  # only reuse on the first iter
                task_lossa = self.loss_func(task_outputa, labela)
                grads = tf.gradients(task_lossa, list(weights.values()))
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(zip(weights.keys(), [weights[key] - self.update_lr*gradients[key] for key in weights.keys()]))
                output = self.forward(inputb, fast_weights, reuse=True)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))

                for j in range(num_updates - 1):
                    loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True), labela)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*gradients[key] for key in fast_weights.keys()]))
                    output = self.forward(inputb, fast_weights, reuse=True)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))
 
                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

                if self.classification:
                    task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1), tf.argmax(labela, 1))
                    for j in range(num_updates):
                        task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1), tf.argmax(labelb, 1)))
                    task_output.extend([task_accuracya, task_accuraciesb])

                return task_output

            if FLAGS.norm is not 'None':
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)
            
            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]
            if self.classification:
                out_dtype.extend([tf.float32, [tf.float32]*num_updates])
            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            if self.classification:
                outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result
            else:
                outputas, outputbs, lossesa, lossesb  = result
                
        ## Performance & Optimization
        if 'train' in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size) # lossesa (25,25) => TensorShape([])
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)] # lossesb[0] (25, 75) => total_losses2[0] TensorShape([])
            # after the map_fn
            self.outputas, self.outputbs = outputas, outputbs
            if self.classification:
                self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)

            if FLAGS.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_updates-1])
                gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
                self.metatrain_op = optimizer.apply_gradients(gvs)
        else:
            self.outputas, self.outputbs = outputas, outputbs
            self.metaval_total_loss_1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_losses_2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            if self.classification:
                self.metaval_total_accuracy_1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                self.metaval_total_accuracies_2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
        # elif 'multi' in prefix:
        #     self.outputas, self.outputbs = outputas, outputbs
        #     self.metaval_total_loss_m1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
        #     self.metaval_total_losses_m2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
        #     if self.classification:
        #         self.metaval_total_accuracy_m1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
        #         self.metaval_total_accuracies_m2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
        # elif 'per' in prefix:
        #     self.outputas, self.outputbs = outputas, outputbs
        #     self.metaval_total_loss_p1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
        #     self.metaval_total_losses_p2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
        #     if self.classification:
        #         self.metaval_total_accuracy_p1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
        #         self.metaval_total_accuracies_p2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
        ## Summaries
        tf.compat.v1.summary.scalar(prefix+'Pre-update loss', total_loss1)
        if self.classification:
            tf.summary.scalar(prefix+'Pre-update accuracy', total_accuracy1)

        for j in range(num_updates):
            tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(prefix+'Post-update accuracy, step ' + str(j+1), total_accuracies2[j])

    ### Network construction functions (fc networks and conv networks)
    def construct_fc_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1,len(self.dim_hidden)):
            weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[i-1], self.dim_hidden[i]], stddev=0.01))
            weights['b'+str(i+1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        weights['w'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        weights['b'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.zeros([self.dim_output]))
        return weights

    def forward_fc(self, inp, weights, reuse=False):
        hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
        for i in range(1,len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w'+str(i+1)]) + weights['b'+str(i+1)], activation=tf.nn.relu, reuse=reuse, scope=str(i+1))
        return tf.matmul(hidden, weights['w'+str(len(self.dim_hidden)+1)]) + weights['b'+str(len(self.dim_hidden)+1)]
    
    def construct_resnet_weights(self):
        weights = {}
        
        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        k = 3
        num_filters = [32, 64, 128]
        
        weights['sc1'] = tf.compat.v1.get_variable('shortcut1', [1, 1, self.channels, num_filters[0]], initializer=conv_initializer, dtype=dtype) # (1,1,16,32)
        weights['conv1'] = tf.compat.v1.get_variable('conv1', [k, k, self.channels, num_filters[0]], initializer=conv_initializer, dtype=dtype)   # (3,3,16,32)
        weights['conv2'] = tf.compat.v1.get_variable('conv2', [k, k, num_filters[0], num_filters[0]], initializer=conv_initializer, dtype=dtype)  # (3,3,32,32)
        
        weights['sc2'] = tf.compat.v1.get_variable('shortcut2', [1, 1, num_filters[0], num_filters[1]])                                           # (1,1,32,64)
        weights['conv3'] = tf.compat.v1.get_variable('conv3', [k, k, num_filters[0], num_filters[1]], initializer=conv_initializer, dtype=dtype)  # (3,3,32,64)
        weights['conv4'] = tf.compat.v1.get_variable('conv4', [k, k, num_filters[1], num_filters[1]], initializer=conv_initializer, dtype=dtype)  # (3,3,64,64)
        
        weights['sc3'] = tf.compat.v1.get_variable('shortcut3', [1, 1, num_filters[1], num_filters[2]])                                           # (1,1,64,128)
        weights['conv5'] = tf.compat.v1.get_variable('conv5', [k, k, num_filters[1], num_filters[2]], initializer=conv_initializer, dtype=dtype)  # (3,3,64,128)
        weights['conv6'] = tf.compat.v1.get_variable('conv6', [k, k, num_filters[2], num_filters[2]], initializer=conv_initializer, dtype=dtype)  # (3,3,128,128)
        
        weights['sc4'] = tf.compat.v1.get_variable('shortcut4', [1, 1, num_filters[2], num_filters[2]])                                           # (1,1,128,128)
        weights['conv7'] = tf.compat.v1.get_variable('conv7', [k, k, num_filters[2], num_filters[2]], initializer=conv_initializer, dtype=dtype)  # (3,3,128,128)
        weights['conv8'] = tf.compat.v1.get_variable('conv8', [k, k, num_filters[2], num_filters[2]], initializer=conv_initializer, dtype=dtype)  # (3,3,128,128)
        
        weights['w9'] = tf.compat.v1.get_variable('w9', [num_filters[2]*8*2, self.dim_output], initializer=fc_initializer) # (8192,6)
        weights['bias'] = tf.Variable(tf.zeros([self.dim_output]), name='bias') # (6, )
        
        return weights
        
    def forward_resnet(self, inp, weights, reuse=False, scope=''):
        stride, no_stride = [1,2,2,1], [1,1,1,1]
        channels = self.channels
        
        inp = tf.reshape(inp, [-1, self.img_size[0], self.img_size[1], channels]) # (30,250,60,16) (,125,30,8)
        
        shortcut1 = tf.nn.conv2d(inp, weights['sc1'], stride, 'SAME')    # (,63,15,32)
        shortcut1 = normalize(shortcut1, None, reuse, scope+'0')         # (,63,15,32)
        conv1 = tf.nn.conv2d(inp, weights['conv1'], no_stride, 'SAME')   # (,125,30,32)
        bn1 = normalize(conv1, tf.nn.relu, reuse, scope+'1')    # (,125,30,32)
        conv2 = tf.nn.conv2d(bn1, weights['conv2'], stride, 'SAME')    # (63,15,32)
        bn2 = normalize(conv2, None, reuse, scope+'2') # (,63,15,32)
        output1 = layers.add([shortcut1, bn2]) # (30,63,15,32)
        tf.nn.relu(output1)
        
        shortcut2 = tf.nn.conv2d(output1, weights['sc2'], stride, 'SAME') # (,32,8,64)
        shortcut2 = normalize(shortcut2, None, reuse, scope+'3') # (,32,8,64)
        conv3 = tf.nn.conv2d(output1, weights['conv3'], no_stride, 'SAME')# (,63,15,64)
        bn3 = normalize(conv3, tf.nn.relu, reuse, scope+'4') # (,63,15,64)
        conv4 = tf.nn.conv2d(bn3, weights['conv4'], stride, 'SAME') # (,32,8,64)
        bn4 = normalize(conv4, None, reuse, scope+'5') # (,32,8,64)
        output2 = layers.add([shortcut2, bn4]) # (,32,8,64)
        tf.nn.relu(output2)
        
        shortcut3 = tf.nn.conv2d(output2, weights['sc3'], stride, 'SAME') # (,16,4,128)
        shortcut3 = normalize(shortcut3, None, reuse, scope+'6') # (,16,4,128)
        conv5 = tf.nn.conv2d(output2, weights['conv5'], no_stride, 'SAME') # (,32,8,128)
        bn5 = normalize(conv5, tf.nn.relu, reuse, scope+'7') # (,32,8,128)
        conv6 = tf.nn.conv2d(bn5, weights['conv6'], stride, 'SAME') # (,16,4,128)
        bn6 = normalize(conv6, None, reuse, scope+'8') # (,16,4,128)
        output3 = layers.add([shortcut3, bn6]) # (,16,4,128)
        tf.nn.relu(output3)
        
        shortcut4 = tf.nn.conv2d(output3, weights['sc4'], no_stride, 'SAME') # (,16,4,128)
        shortcut4 = normalize(shortcut4, None, reuse, scope+'9') # (,16,4,128)
        conv7 = tf.nn.conv2d(output3, weights['conv7'], no_stride, 'SAME') # (,16,4,128)
        bn7 = normalize(conv7, tf.nn.relu, reuse, scope+'10') # (,16,4,128)
        conv8 = tf.nn.conv2d(bn7, weights['conv8'], no_stride, 'SAME') # (,16,4,128)
        bn8 = normalize(conv8, None, reuse, scope+'11') # (,16,4,128)
        output4 = layers.add([shortcut4, bn8]) # (,16,4,128)
        tf.nn.relu(output4)
        
        output = tf.nn.avg_pool(output4, stride, stride, 'VALID') # (,8,2,128)
        fc_input = tf.reshape(output, [-1, np.prod([int(dim) for dim in output.get_shape()[1:]])]) # (30,8192) 4096->8192->2048
        
        return tf.matmul(fc_input, weights['w9']) + weights['bias'] # (30,6)
        
    def construct_conv_weights(self):
        weights = {}

        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        k = 3  

        # 卷积层参数
        weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv5'] = tf.get_variable('conv5', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b5'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv6'] = tf.get_variable('conv6', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b6'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv7'] = tf.get_variable('conv7', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b7'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv8'] = tf.get_variable('conv8', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b8'] = tf.Variable(tf.zeros([self.dim_hidden]))
        
        # 全连接层参数
        weights['w9'] = tf.get_variable('w9', [self.dim_hidden*15, self.dim_output], initializer=fc_initializer)
        weights['b9'] = tf.Variable(tf.zeros([self.dim_output]), name='b9')
        
        return weights

    def forward_conv(self, inp, weights, reuse=False, scope=''):
        # reuse is for the normalization parameters.
        stride, no_stride = [1,2,2,1], [1,1,1,1]
        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size[0], self.img_size[1], channels]) # (30,250,30,16)

        conv_output1 = tf.nn.conv2d(inp, weights['conv1'], no_stride, 'SAME') + weights['b1'] # (30,250,30,32)
        conv_output2 = tf.nn.conv2d(conv_output1, weights['conv2'], no_stride, 'SAME') + weights['b2'] # (30,250.30,32)
        normed1 = normalize(conv_output2, tf.nn.relu, reuse, scope+'0') # (30,250,30,32)
        normed1 = tf.nn.max_pool(normed1, stride, stride, 'VALID') # (30,125,15,32)
        conv_output3 = tf.nn.conv2d(normed1, weights['conv3'], no_stride, 'SAME') + weights['b3'] # (30,125,15,32)
        conv_output4 = tf.nn.conv2d(conv_output3, weights['conv4'], no_stride, 'SAME') + weights['b4'] # (30,125,15,32)
        normed2 = normalize(conv_output4, tf.nn.relu, reuse, scope+'1')
        normed2 = tf.nn.max_pool(normed2, stride, stride, 'VALID') # (30,62,7,32)
        conv_output5 = tf.nn.conv2d(normed2, weights['conv5'], no_stride, 'SAME') + weights['b5'] # (30,62,7,32)
        conv_output6 = tf.nn.conv2d(conv_output5, weights['conv6'], no_stride, 'SAME') + weights['b6']
        normed3 = normalize(conv_output6, tf.nn.relu, reuse, scope+'2')
        normed3 = tf.nn.max_pool(normed3, stride, stride, 'VALID') # (30,31,3,32)
        conv_output7 = tf.nn.conv2d(normed3, weights['conv7'], no_stride, 'SAME') + weights['b7']
        conv_output8 = tf.nn.conv2d(conv_output7, weights['conv8'], no_stride, 'SAME') + weights['b8']
        normed3 = normalize(conv_output8, tf.nn.relu, reuse, scope+'3')
        normed3 = tf.nn.max_pool(normed3, stride, stride, 'VALID') # (30,15,1,32)
        
        fc_input = tf.reshape(normed3, [-1, np.prod([int(dim) for dim in normed3.get_shape()[1:]])]) # (30,480)
        return tf.matmul(fc_input, weights['w9']) + weights['b9'] # (30,6)


