from abc import ABC, abstractmethod
import numpy as np
from numpy.core.numerictypes import ScalarType
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection 
from sklearn.preprocessing import Binarizer

class DataSet(ABC):
    @abstractmethod
    def get_train(self):
        pass

    @abstractmethod
    def get_test(self):
        pass

    @abstractmethod
    def get_val(self):
        pass


class MNIST(DataSet):

    def __init__(self, normalize1=0, method='svd', comps=10, val_size=1000, seed=9) -> None:
        self.rnd = np.random.RandomState(seed)
        self.m1 = None
        self.sigma1 =None
        self.m2 = None
        self.sigma2 =None
        self.V = None
        self.U = None
        self.components = comps
        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = np.array(x_train / 255.0, np.float32), np.array(x_test / 255.0, np.float32)
        x_train_temp=x_train.reshape((60000,-1))
        x_test_temp=x_test.reshape((10000,-1))
        self.sigma_tresh=1e-4
        # standarization
        if normalize1: 
            self.m1,self.sigma1=np.mean(x_train_temp,axis=0),np.std(x_train_temp,axis=0)   ## for the first normal layer 
            self.sigma1[self.sigma1<self.sigma_tresh]=1 
            x_train_temp=(x_train_temp-self.m1)/self.sigma1
        #method='svd'
        if method=='svd':
            M = np.dot(x_train_temp[:60000].T,x_train_temp[:60000])
            self.U, s, self.V =np.linalg.svd(M)
 
        x_proj=np.dot(x_train_temp,self.V[:self.components,:].T)         
        self.m2=  np.mean(x_proj,axis=0)
        self.sigma2= np.std(x_proj,axis=0) 
        self.sigma2[self.sigma2<self.sigma_tresh]=1 
        
        self.x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
        self.y_train = np.array(y_train, np.int64)
        self.y_test = np.array(y_test, np.int64)
        self.x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
        self.x_train, self.y_train, self.x_val, self.y_val = self.split_data(self.rnd, val_size // 10, self.x_train, self.y_train)
 
 
    def split_data(self, rnd, sample_per_class, x, y):
        x_equalized = ()
        x_remained = ()
        y_equalized = ()
        y_remained = ()
        for i in np.unique(y):
            idxs = rnd.permutation(np.sum(y == i))
            x_i = x[y == i]
            y_i = y[y == i]
            x_equalized = x_equalized + (x_i[idxs[:sample_per_class]],)
            y_equalized = y_equalized + (y_i[idxs[:sample_per_class]],)
            x_remained = x_remained + (x_i[idxs[sample_per_class:]],)
            y_remained = y_remained + (y_i[idxs[sample_per_class:]],)
        return np.concatenate(x_remained, axis=0), np.concatenate(y_remained, axis=0), \
               np.concatenate(x_equalized, axis=0), np.concatenate(y_equalized, axis=0)

    def get_bound(self):
        return (0., 1.)

    def get_input_shape(self):
        return self.x_train.shape[1:]

    def get_nb_classes(self):
        return np.unique(self.y_train).shape[0]
    
    def get_nb_components(self):
        return self.components

    def get_v(self):
        return self.V
    
    def get_u(self):
        return self.U
    
    def get_mean1(self):
        return self.m1
    def get_sigma1(self):
        return self.sigma1
    def get_mean2(self):
        return self.m2
    def get_sigma2(self):
        return self.sigma2
    
    def get_train(self):
        return self.x_train, self.y_train

    def get_test(self):
        return self.x_test, self.y_test

    def get_val(self):
        return self.x_val, self.y_val

    def get_name(self):
        return 'MNIST'
    
    
class FMNIST(DataSet):

    def __init__(self, normalize1=0, method='svd', comps=10, val_size=1000, seed=9) -> None:
        self.rnd = np.random.RandomState(seed)
        self.m1 = None
        self.sigma1 =None
        self.m2 = None
        self.sigma2 =None
        self.V = None
        self.U = None
        self.components = comps
        fmnist = tf.keras.datasets.fashion_mnist

        (x_train, y_train), (x_test, y_test) = fmnist.load_data()
        x_train, x_test = np.array(x_train / 255.0, np.float32), np.array(x_test / 255.0, np.float32) 
        
        x_train_temp=x_train.reshape((60000,-1))
        x_test_temp=x_test.reshape((10000,-1))
        self.sigma_tresh=1e-4
        # standarization
        if normalize1: 
            self.m1,self.sigma1=np.mean(x_train_temp,axis=0),np.std(x_train_temp,axis=0)   ## for the first normal layer 
            self.sigma1[self.sigma1<self.sigma_tresh]=1 
            x_train_temp=(x_train_temp-self.m1)/self.sigma1
        #method='svd'
        if method=='svd':
            M = np.dot(x_train_temp[:60000].T,x_train_temp[:60000])
            self.U, s, self.V =np.linalg.svd(M)
        
        x_proj=np.dot(x_train_temp,self.V[:self.components,:].T)
                   
        self.m2=  np.mean(x_proj,axis=0)
        self.sigma2= np.std(x_proj,axis=0) 
        self.sigma2[self.sigma2<self.sigma_tresh]=1 
        
        self.x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
        self.y_train = np.array(y_train, np.int64)
        self.y_test = np.array(y_test, np.int64)
        self.x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
        self.x_train, self.y_train, self.x_val, self.y_val = self.split_data(self.rnd, val_size // 10, self.x_train, self.y_train)
 
 
    def split_data(self, rnd, sample_per_class, x, y):
        x_equalized = ()
        x_remained = ()
        y_equalized = ()
        y_remained = ()
        for i in np.unique(y):
            idxs = rnd.permutation(np.sum(y == i))
            x_i = x[y == i]
            y_i = y[y == i]
            x_equalized = x_equalized + (x_i[idxs[:sample_per_class]],)
            y_equalized = y_equalized + (y_i[idxs[:sample_per_class]],)
            x_remained = x_remained + (x_i[idxs[sample_per_class:]],)
            y_remained = y_remained + (y_i[idxs[sample_per_class:]],)
        return np.concatenate(x_remained, axis=0), np.concatenate(y_remained, axis=0), \
               np.concatenate(x_equalized, axis=0), np.concatenate(y_equalized, axis=0)

    def get_bound(self):
        return (0., 1.)

    def get_input_shape(self):
        return self.x_train.shape[1:]

    def get_nb_classes(self):
        return np.unique(self.y_train).shape[0]
    
    def get_nb_components(self):
        return self.components

    def get_v(self):
        return self.V
    
    def get_u(self):
        return self.U
    
    def get_mean1(self):
        return self.m1
    def get_sigma1(self):
        return self.sigma1
    def get_mean2(self):
        return self.m2
    def get_sigma2(self):
        return self.sigma2
    
    def get_train(self):
        return self.x_train, self.y_train

    def get_test(self):
        return self.x_test, self.y_test

    def get_val(self):
        return self.x_val, self.y_val

    def get_name(self):
        return 'FMNIST'

     
 