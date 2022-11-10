import csv
import numpy as np
import random
import tensorflow as tf

from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_integer('num_classes', 6, 'number of classes used in classification')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 601, 'number of metatraining iterations.')
flags.DEFINE_integer('meta_batch_size', 64, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 4, 'number of examples used for inner gradient update (K for K-shot learning).') # 5
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.')
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
flags.DEFINE_integer('icd_count', 1, '1')
flags.DEFINE_integer('all_icd_count', 5, '5 for jxb, 10 for all.')
flags.DEFINE_integer('num_samples_per_class', 10, '10')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 32, 'number of filters for conv nets')
flags.DEFINE_bool('conv', False, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('resnet', True, 'whether or not to use a resnet')
flags.DEFINE_bool('max_pool', True, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', False, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', 'logs'+str(FLAGS.update_batch_size), 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', False, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', True, 'Set to true to test on the the test set, False for the validation set.')

flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)')

def train(model, saver, sess, exp_string, data_generator, resume_itr=1):
    SUMMARY_INTERVAL = 1 # 100
    SAVE_INTERVAL = 100 # 1000

    PRINT_INTERVAL = 1 #100
    TEST_PRINT_INTERVAL = 20

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []
    preacc, postacc = [], []

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    # multitask_weights, reg_weights = [], []

    wn_train_acc = []
    wn_val_acc = []
    # wn_val_acc_p = []

    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        feed_dict = {}
        if 'generate' in dir(data_generator):
            batch_x, batch_y, amp, phase = data_generator.generate()

            inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :]
            labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb}

        if itr < FLAGS.pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:
            input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
            input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])

        result = sess.run(input_tensors, feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            preacc.append(result[-2]) # acc_a
            prelosses.append(result[2]) # losses_a
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[3]) # losses_b
            postacc.append(result[-1]) # acc_b

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses)) + '; ' + str(np.mean(preacc)) + ', ' + str(np.mean(postacc))
            print(print_str)
            wn_train_acc.append([itr, np.mean(preacc), np.mean(postacc), np.mean(prelosses), np.mean(postlosses)])
            prelosses, postlosses = [], []
            preacc, postacc = [],[]

        if (itr!=0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:
            input_tensors = [model.metaval_total_accuracy_1, model.metaval_total_accuracies_2[FLAGS.num_updates-1], model.metaval_total_loss_1, model.metaval_total_losses_2[FLAGS.num_updates-1]]
            result = sess.run(input_tensors, feed_dict)
            print('Validation results(m): ' + str(result[2]) + ', ' + str(result[3]) + '; ' + str(result[0]) + ', ' + str(result[1]))                                                                                                                                     
            wn_val_acc.append([itr, result[0], result[1], result[2], result[3]])
    
    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))
    
    
    wn_train_acc = np.array(wn_train_acc)
    wn_val_acc = np.array(wn_val_acc)
    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    
    out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'train_ubs' + str(FLAGS.update_batch_size)+ '_itr' + str(FLAGS.metatrain_iterations-1) + '.csv'
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(wn_train_acc[:,0])
        writer.writerow(wn_train_acc[:,1])
        writer.writerow(wn_train_acc[:,2])
        writer.writerow(wn_train_acc[:,3])
        writer.writerow(wn_train_acc[:,4])
    out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'val_ubs' + str(FLAGS.update_batch_size)+ '_itr' + str(FLAGS.metatrain_iterations-1) + '.csv'
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(wn_val_acc[:,0])
        writer.writerow(wn_val_acc[:,1])
        writer.writerow(wn_val_acc[:,2])
        writer.writerow(wn_val_acc[:,3])
        writer.writerow(wn_val_acc[:,4])

NUM_TEST_POINTS = 100

def test(model, saver, sess, exp_string, data_generator, test_num_updates=None):
    num_classes = data_generator.num_classes # for classification, 1 otherwise

    np.random.seed(1)
    random.seed(1)

    metaval_accuracies, metaval_losses = [], []
    cm_data = []

    for itr in range(NUM_TEST_POINTS):
        if 'generate' not in dir(data_generator):
            feed_dict = {}
            feed_dict = {model.meta_lr : 0.0}
        else:
            batch_x, batch_y, amp, phase = data_generator.generate(train=False)

            inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            inputb = batch_x[:,num_classes*FLAGS.update_batch_size:, :]
            labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            labelb = batch_y[:,num_classes*FLAGS.update_batch_size:, :]

            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}

        if model.classification:
            result = sess.run([[model.metaval_total_accuracy_1] + model.metaval_total_accuracies_2, [model.metaval_total_loss_1] + model.metaval_total_losses_2, model.labelb, model.outputbs], feed_dict)
            label = sess.run(tf.argmax(result[2][0],1)) #sess.run(tf.argmax(model.labelb[0],1))
            pred = sess.run(tf.argmax(tf.nn.softmax(result[3][-1][0]),1)) #sess.run(tf.argmax(tf.nn.softmax(model.outputbs[-1][0]),1))
            cm_data.append(label)
            cm_data.append(pred)

        else:  # this is for sinusoid
            result = sess.run([model.total_loss1] +  model.total_losses2, feed_dict)
        metaval_accuracies.append(result[0])
        metaval_losses.append(result[1])
        print(str(itr)+":"+str(result[0]))
    
    filename = FLAGS.logdir + '/' + exp_string + '/cm_data.csv'
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(cm_data)

    filename = FLAGS.logdir + '/' + exp_string + '/test.csv'
    with open(filename, 'w',newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update'+str(i) for i in range(31)])
        writer.writerows(metaval_accuracies)

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)
    metaval_losses = np.array(metaval_losses)
    means_loss = np.mean(metaval_losses, 0)
    
    print('Mean validation accuracy/loss, stddev, and confidence intervals, MULTI')
    print((means, stds, ci95))

    out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '.csv'
    #out_filename = FLAGS.logdir + '/_' + exp_string + '/test_pre.csv'
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update'+str(i) for i in range(len(means))])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)
        writer.writerow(means_loss)


def main():
    
    if FLAGS.train == True:
        test_num_updates = 1
    else:
        test_num_updates = 10

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1
    
    if FLAGS.metatrain_iterations == 0:
        assert FLAGS.meta_batch_size == 1
        assert FLAGS.update_batch_size == 1
        data_generator = DataGenerator(1, FLAGS.meta_batch_size)
    else:
        data_generator = DataGenerator(FLAGS.num_samples_per_class, FLAGS.meta_batch_size)


    dim_output = data_generator.dim_output
    dim_input = data_generator.dim_input

    tf_data_load = True
    num_classes = data_generator.num_classes

    if FLAGS.train: # only construct training model if needed
        random.seed(5)
        image_tensor, label_tensor = data_generator.make_data_tensor()
        
        inputa = tf.slice(image_tensor, [0,0,0,0,0], [-1,num_classes*FLAGS.update_batch_size,-1,-1,-1])
        inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size,0,0,0], [-1,-1,-1,-1,-1])
        labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
        labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
        input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

    random.seed(8)
    image_tensor, label_tensor = data_generator.make_test_data_tensor(FLAGS.icd_count)
    inputa = tf.slice(image_tensor, [0,0,0,0,0], [-1,num_classes*FLAGS.update_batch_size*FLAGS.icd_count,-1,-1,-1])
    inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size*FLAGS.icd_count,0,0,0], [-1,-1,-1,-1,-1])
    labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size*FLAGS.icd_count, -1])
    labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size*FLAGS.icd_count, 0], [-1,-1,-1])
    metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb} # 测试
    
    random.seed()

    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    if FLAGS.train or not tf_data_load:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    if tf_data_load:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval')
    model.summ_op = tf.compat.v1.summary.merge_all()

    saver = loader = tf.compat.v1.train.Saver(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=3)

    sess = tf.InteractiveSession()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size)# + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr)

    resume_itr = 1
    model_file = None

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)   
        print('============')
        print('model_file:', model_file)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    if FLAGS.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        test(model, saver, sess, exp_string, data_generator, test_num_updates)
    print("Over!")

if __name__ == "__main__":
    print("hello,world!")
    main()
