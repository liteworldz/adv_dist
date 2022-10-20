from argparse import ArgumentParser
import os
import tensorflow as tf
import numpy as np
from dataset import MNIST, FMNIST
from models import SampleCNN, MLP
import util
import time
from foolbox.attacks import LinfPGD
from foolbox.models import TensorFlowModel


def batch_attack(imgs, labels, attack, fmodel, eps, batch_size):
    adv = []
    for i in range(int(np.ceil(imgs.shape[0] / batch_size))):
        x_adv, _, success = attack(fmodel, imgs[i * batch_size:(i + 1) * batch_size],
                                   criterion=labels[i * batch_size:(i + 1) * batch_size], epsilons=eps)
        adv.append(x_adv)
    return np.concatenate(adv, axis=0)


def main(params):
    comps = params.comps 
    dataset = params.dataset
    if dataset == 'MNIST':
        ds = MNIST(params.normalize1, params.method, comps) 
    elif dataset == 'FMNIST':
        ds = FMNIST(params.normalize1, params.method, comps) 
    x_train, y_train = ds.get_train()
    x_val, y_val = ds.get_val()
    print(x_train.shape, np.bincount(y_train), x_val.shape, np.bincount(y_val))
    
    model_holder =  MLP()
    model = model_holder.build_model(ds.get_input_shape(), ds.get_nb_classes(), ds.get_nb_components(), ds.get_mean1(), ds.get_sigma1(), ds.get_mean2(), ds.get_sigma2(), params.normalize1, params.normalize2, params.freeze, params.denses, params.dense_size)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['sparse_categorical_accuracy']
    ws=model.get_weights()
    V = ds.get_v() 
    scaler=0 
    ws[0]=  V[:ds.get_nb_components(),:].T
    if params.freeze:
     model.set_weights(ws)
    if params.freeze: 
        if params.normalize1:
            model.layers[2].trainable=False
        else:
            model.layers[1].trainable=False
    model.compile(tf.keras.optimizers.Adam(1e-4), loss_fn, metrics)
    m_path = os.path.join(params.save_dir, model_holder.get_name())
    util.mk_parent_dir(m_path) 
    
    label = '_model_comps_' + str(ds.get_nb_components()) + '_dataset_' +  params.dataset +  '_method_' + params.method + '_normalized1_' +  str(params.normalize1) + '_normalized2_' +  str(params.normalize2) + '_freezed_' +  str(params.freeze) + '_denses_' +  str(params.denses) + '_dense-size_' +  str(params.dense_size)
    callbacks = [tf.keras.callbacks.ModelCheckpoint(m_path + label  + '_{epoch:03d}.h5'),
                 tf.keras.callbacks.CSVLogger(os.path.join(params.save_dir, label + '.csv'))]

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(seed=9, buffer_size=x_train.shape[0], reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(params.batch_size)

    attack = LinfPGD(abs_stepsize=params.step_size, steps=params.steps, random_start=True)
    fmodel = TensorFlowModel(model, bounds=(0, 1), device='/device:GPU:0')
    imgs_val, labels_val = tf.convert_to_tensor(x_val), tf.convert_to_tensor(y_val)
    for cb in callbacks:
        cb.set_model(model)
        cb.set_params(
            {'batch_size': params.batch_size, 'epochs': params.epoch, 'steps': x_train.shape[0] // params.batch_size,
             'samples': x_train.shape[0], 'verbose': 0,
             'do_validation': True,
             'metrics': ['loss', 'accuracy', 'val_loss', 'val_accuracy']})
        cb.on_train_begin()
    for i in range(params.epoch):
        for cb in callbacks:
            cb.on_epoch_begin(i)
        delta0 = time.time()
        a_loss, a_acc = 0, 0
        for b_idx, (x_batch, y_batch) in enumerate(train_dataset):
            print('\r', b_idx, end=' ')
            x_adv_batch, _, success = attack(fmodel, x_batch,
                                             criterion=y_batch, epsilons=params.eps)
            batch_eval = model.train_on_batch(x_adv_batch, y_batch)
            a_loss = a_loss + batch_eval[0]
            a_acc = a_acc + batch_eval[1]
        x_adv_val = batch_attack(imgs_val, labels_val, attack, fmodel, params.eps, params.batch_size)
        train_eval = [a_loss / (b_idx + 1), a_acc / (b_idx + 1)]
        val_eval = model.evaluate(x_adv_val, y_val, verbose=0)
        stats = {'loss': train_eval[0], 'accuracy': train_eval[1], 'val_loss': val_eval[0],
                 'val_accuracy': val_eval[1]}
        print(i, time.time() - delta0, 's', stats)
        for cb in callbacks:
            cb.on_epoch_end(i, stats)
            if i == (params.epoch - 1) or model.stop_training:
                cb.on_train_end()
        if model.stop_training:
            break


if __name__ == '__main__':
    parser = ArgumentParser(description='Main entry point')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--comps", type=int, default=100)
    parser.add_argument("--dataset", type=str, default='MNIST') 
    parser.add_argument("--method", type=str, default='svd')
    parser.add_argument("--normalize1", type=int, default=0)
    parser.add_argument("--normalize2", type=int, default=0)
    parser.add_argument("--freeze", type=int, default=0)  
    parser.add_argument("--denses", type=int, default=1)   
    parser.add_argument("--dense_size", type=int, default=256)
    parser.add_argument("--memory_limit", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default=os.path.join('saved_models'))
    parser.add_argument('--step_size', type=float, default=0.025)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--eps', type=float, default=0.1)
    #step_size=eps/steps*2.5
    #eps=0.1
    #steps=10 --> 30 min
    # step_size=0.1/10*2.5=0.025
    # steps_size/2.5*steps=0.1
    # 0.025/2.5*10
    #steps=200 --> 600 min --> 10hour
    #steps=40 --> 120 min
    
    FLAGS = parser.parse_args()
    np.random.seed(9)
    if FLAGS.gpu is not None:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        selected = gpus[FLAGS.gpu]
        tf.config.experimental.set_visible_devices(selected, 'GPU')
        tf.config.experimental.set_memory_growth(selected, True)
        tf.config.experimental.set_virtual_device_configuration(
            selected,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=FLAGS.memory_limit)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        l_gpu = logical_gpus[0]
        with tf.device(l_gpu.name):
            main(FLAGS)
    else:
        main(FLAGS)
