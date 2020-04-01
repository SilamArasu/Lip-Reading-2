import sys, os
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_PATH)

from videos import Video
from aligns import Align
from keras import backend as K
import numpy as np
import keras
import pickle
import glob
import multiprocessing
import threading
from keyframe import KeyFrame
from sklearn.utils import shuffle
import random
from vidaug import select_augmentation

class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)

def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

def get_list(l, index, size):
    ret = l[index:index+size]
    while size - len(ret) > 0:
        ret += l[0:size - len(ret)] # It goes circular
    return ret

# datasets/[train|val]/<sid>/<id>/<image>.png
# or datasets/[train|val]/<sid>/<id>.mpg
# datasets/align/<id>.align
random_seed = 17

class Generator(keras.callbacks.Callback):
    def __init__(self, dataset_path, minibatch_size, img_c, img_w, img_h, frames_n, absolute_max_string_len=32):
        self.dataset_path   = dataset_path
        self.minibatch_size = minibatch_size
        self.img_c          = img_c
        self.img_w          = img_w
        self.img_h          = img_h
        self.frames_n       = frames_n
        self.absolute_max_string_len = absolute_max_string_len
        self.train_index = multiprocessing.Value('i', 0)    
        self.val_index   = multiprocessing.Value('i', 0)
        self.train_epoch  = multiprocessing.Value('i', -1)
        self.keyframe       = KeyFrame()
        self.train_path     = os.path.join(self.dataset_path, 'train')
        self.val_path       = os.path.join(self.dataset_path, 'val')
        self.align_path     = os.path.join(self.dataset_path, 'align')
        self.build_dataset()

    @property
    def training_size(self):
        return len(self.train_list)

    @property
    def default_training_steps(self):
        return self.training_size / self.minibatch_size

    @property
    def validation_size(self):
        return len(self.val_list)

    @property
    def default_validation_steps(self):
        return self.validation_size / self.minibatch_size

    def prepare_vidlist(self, path):
        video_list = []
        l = None
        for video_path in glob.glob(path):
            l = len(os.listdir(video_path))
            if l == 75:
                video_list.append(video_path)
            else:
                print("Error loading video: "+video_path+" less than 75 frames("+str(l)+")")

        return video_list

    def prepare_align(self, video_list):
        align_dict = {}
        for video_path in video_list:
            video_id = os.path.splitext(video_path)[0].split('/')[-1]
            align_path = os.path.join(self.align_path, video_id)+".align"
            align_dict[video_id] = Align(self.absolute_max_string_len).from_file(align_path)
        return align_dict

    def build_dataset(self):
        print("\nLoading datas...")
        self.train_list = self.prepare_vidlist(os.path.join(self.train_path, '*', '*'))
        self.val_list   = self.prepare_vidlist(os.path.join(self.val_path, '*', '*'))
        self.align_dict = self.prepare_align(self.train_list + self.val_list)
        self.steps_per_epoch  = self.default_training_steps
        print("Number of training videos = ", self.training_size)
        print("Number of validation videos = ", self.validation_size)
        print("Steps per epoch ", round(self.steps_per_epoch))
        print("")

        np.random.shuffle(self.train_list)

    def get_batch(self, index, size, train):
        if train:
            video_list = self.train_list
        else:
            video_list = self.val_list

        X_data_path = get_list(video_list, index, size)
        X_data = []
        Y_data = []
        label_length = []
        input_length = []
        source_str = []
        for path in X_data_path:
            video = Video().from_frames(path)
            align = self.align_dict[path.split('/')[-1]]
            if train == True:
                video= self.keyframe.extract(video)
                video.data = select_augmentation(video.data)

            X_data.append(video.data)
            Y_data.append(align.padded_label)
            label_length.append(align.label_length) 
            input_length.append(video.length) 
            source_str.append(align.sentence)

        source_str = np.array(source_str)
        label_length = np.array(label_length)
        input_length = np.array(input_length)
        Y_data = np.array(Y_data)
        X_data = np.array(X_data).astype(np.float32) / 255 
        X_data, Y_data, input_length, label_length, source_str = shuffle(X_data, Y_data, input_length, label_length, source_str, random_state=random_seed)

        inputs = {'the_input': X_data,
                  'the_labels': Y_data,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str 
                  }
        outputs = {'ctc': np.zeros([size])}  
        return (inputs, outputs)

    @threadsafe_generator
    def next_train(self):
        while True:
            with self.train_index.get_lock(), self.train_epoch.get_lock():
                train_index = self.train_index.value
                self.train_index.value += self.minibatch_size
                if train_index >= self.steps_per_epoch * self.minibatch_size:
                    train_index = 0
                    self.train_epoch.value += 1
                    self.train_index.value = self.minibatch_size
                if self.train_epoch.value < 0:
                    self.train_epoch.value = 0
                if self.train_index.value >= self.training_size:
                    self.train_index.value = self.train_index.value % self.minibatch_size
            ret = self.get_batch(train_index, self.minibatch_size, train=True)
            yield ret

    @threadsafe_generator
    def next_val(self):
        while True:
            with self.val_index.get_lock():
                val_index = self.val_index.value
                self.val_index.value += self.minibatch_size
                if self.val_index.value >= self.validation_size:
                    self.val_index.value = self.val_index.value % self.minibatch_size
            ret = self.get_batch(val_index, self.minibatch_size, train=False)
            yield ret

    def on_train_begin(self, logs={}):
        with self.train_index.get_lock():
            self.train_index.value = 0
        with self.val_index.get_lock():
            self.val_index.value = 0