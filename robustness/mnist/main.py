from argparse import ArgumentParser
import tensorflow as tf
from tensorflow import keras
from foolbox.attacks import LinfPGD, L2PGD, L1PGD, BoundaryAttack, L2BrendelBethgeAttack, L2DeepFoolAttack
from foolbox.models import TensorFlowModel
import numpy as np
import os
from tensorflow.keras import backend as K


class NormalizingLayer(keras.layers.Layer):
    def __init__(self, mean, sigma, trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.mean = K.constant(mean, dtype=K.floatx())
        self.sigma = K.constant(sigma, dtype=K.floatx())

    def get_config(self):
        base_conf = super().get_config()
        return {**base_conf,
                'mean': np.asfarray(self.mean),
                'sigma': np.asfarray(self.sigma)
                }

    def call(self, inputs, **kwargs):
        out = (inputs - self.mean) / self.sigma   # standarization
        return out


def batch_attack(imgs, labels, attack, fmodel, eps, batch_size):
    adv = []
    for i in range(int(np.ceil(imgs.shape[0] / batch_size))):
        x_adv, _, success = attack(fmodel, imgs[i * batch_size:(i + 1) * batch_size],
                                   criterion=labels[i * batch_size:(i + 1) * batch_size], epsilons=eps)
        adv.append(x_adv)
    return np.concatenate(adv, axis=0)


def main(params):
    print(params)
    netname = params.fname

    if params.dataset == 'MNIST':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif params.dataset == 'FMNIST':
        (x_train, y_train), (x_test,
                             y_test) = tf.keras.datasets.fashion_mnist.load_data()
    # it is important to cast the dtype otherwise the attack will throw exception
    x_test = np.array(x_test.reshape(
        (x_test.shape[0], 28, 28, 1)) / 255., np.float32)
    y_test = np.array(y_test, np.int64)
    if netname.endswith(".h5"):
        model = keras.models.load_model(netname, custom_objects={
                                        'NormalizingLayer': NormalizingLayer})
    else:
        raise Exception(
            "Network filename format is not supported: {0}".format(netname))

    pred = model.predict(x_test)
    
    if FLAGS.attack == 'pgd':
        if FLAGS.norm == "linf":
            attack = LinfPGD(abs_stepsize=params.eps / params.steps *
                             2.5, steps=params.steps, random_start=True)
        elif FLAGS.norm == "l2":
            attack = L2PGD(abs_stepsize=params.eps / params.steps *
                           2.5, steps=params.steps, random_start=True)
        elif FLAGS.norm == "l1":
            attack = L1PGD(abs_stepsize=params.eps / params.steps * 2.5, steps=params.steps, random_start=True)
        else:
            raise Exception('Unknown norm: {0}'.format(FLAGS.norm))
        if params.trials > 1:
            attack = attack.repeat(params.trials)
    elif FLAGS.attack == "ba":
        attack = BoundaryAttack(init_attack=None,#L2DeepFoolAttack(steps=params.steps, candidates=10,overshoot=0.02),
                                steps=params.steps*params.trials, spherical_step=0.01, source_step=0.01, source_step_convergance=1e-07, step_adaptation=1.5, tensorboard=False, update_stats_every_k=10)
    elif FLAGS.attack == "br":
        attack = L2BrendelBethgeAttack(init_attack=None,#L2DeepFoolAttack(steps=params.steps, candidates=10,overshoot=0.02),
                                       overshoot=1.1,
                                       steps=1000,
                                       lr=1e-3,
                                       lr_decay=0.5,
                                       lr_num_decay=20,
                                       momentum=0.8,
                                       binary_search_steps=10)
    else:
        raise Exception('Unknown attack: {0}'.format(FLAGS.attack))

    fmodel = TensorFlowModel(model, bounds=(0, 1), device='/device:GPU:0')
    imgs = tf.convert_to_tensor(x_test)
    labs = tf.convert_to_tensor(y_test)
    acc = np.mean(np.argmax(pred, axis=1) == y_test)
    x_adv = batch_attack(imgs, labs, attack, fmodel,
                         params.eps, params.batch_size)
    x_pert = (x_adv-x_test)
    if FLAGS.norm == 'l2':
        pert_size = np.linalg.norm(x_pert.reshape(
            x_test.shape[0], -1), axis=1, ord=2)
        tgt_shape = np.array(x_test.shape)
        tgt_shape[1:] = 1
        pert_size = pert_size.reshape(tgt_shape)
        #print(pert_size.shape)
        x_pert = x_pert/np.where(pert_size < 1e-3, 1, pert_size)
        pert_size = np.clip(pert_size, 0, FLAGS.eps)
        x_pert = x_pert*pert_size
        x_adv = x_test+x_pert
        pert_size = np.linalg.norm(
            (x_adv-x_test).reshape(x_test.shape[0], -1), axis=1, ord=2)
        #print(pert_size.shape)
        #print(np.min(pert_size), np.mean(pert_size), np.max(pert_size))

    # calculate
    p_adv = model.predict(x_adv)
    a_acc = np.mean(np.argmax(p_adv, axis=1) == y_test)
    # print(netname)
    print('acc=', acc, ',robust-acc=', a_acc)


if __name__ == '__main__':
    parser = ArgumentParser(description='Model evaluation')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--dataset", type=str, default='MNIST')
    parser.add_argument('--fname', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--norm', type=str, default='linf')
    parser.add_argument('--attack', type=str, default='pgd')
    FLAGS = parser.parse_args()
    np.random.seed(9)
    if FLAGS.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    main(FLAGS)
