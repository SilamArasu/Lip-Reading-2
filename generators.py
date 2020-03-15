import sys, os
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_PATH)

#from helpers import text_to_labels
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

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
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
class BasicGenerator(keras.callbacks.Callback):
    def __init__(self, dataset_path, minibatch_size, img_c, img_w, img_h, frames_n, absolute_max_string_len=32, **kwargs):
        self.dataset_path   = dataset_path
        self.minibatch_size = minibatch_size
        self.img_c          = img_c
        self.img_w          = img_w
        self.img_h          = img_h
        self.frames_n       = frames_n
        self.absolute_max_string_len = absolute_max_string_len
        self.cur_train_index = multiprocessing.Value('i', 0)    # Data can be stored in a shared memory using Value
        self.cur_val_index   = multiprocessing.Value('i', 0)
        self.keyframe      = KeyFrame()
        self.random_seed     = 17
        # self.vtype               = kwargs.get('vtype', 'mouth')
        # self.steps_per_epoch     = kwargs.get('steps_per_epoch', None)
        # self.validation_steps    = kwargs.get('validation_steps', None)
        # Process epoch is used by non-training generator (e.g: validation)
        # because each epoch uses different validation data enqueuer
        # Will be updated on epoch_begin
        self.process_epoch       = -1
        # Maintain separate process train epoch because fit_generator only use
        # one enqueuer for the entire training, training enqueuer can contain
        # max_q_size batch data ahead than the current batch data which might be
        # in the epoch different with current actual epoch
        # Will be updated on next_train()
        self.shared_train_epoch  = multiprocessing.Value('i', -1)
        self.process_train_epoch = -1
        self.process_train_index = -1
        self.process_val_index   = -1

    def build(self, **kwargs):
        self.train_path     = os.path.join(self.dataset_path, 'train')
        self.val_path       = os.path.join(self.dataset_path, 'val')
        self.align_path     = os.path.join(self.dataset_path, 'align')
        self.build_dataset()
        # Set steps to dataset size if not set
        self.steps_per_epoch  = self.default_training_steps 
        # if self.steps_per_epoch is None else self.steps_per_epoch
        # self.validation_steps = self.default_validation_steps if self.validation_steps is None else self.validation_steps
        return self

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

        print("Found {} videos for training.".format(self.training_size))
        print("Found {} videos for validation.".format(self.validation_size))
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
            video_unpadded_length = video.length
            # if self.curriculum is not None:
            if train == True:
                video, align, video_unpadded_length = self.keyframe.apply(video, align)

            X_data.append(video.data)
            Y_data.append(align.padded_label)
            label_length.append(align.label_length) # CHANGED [A] -> A, CHECK!
            # input_length.append([video_unpadded_length - 2]) # 2 first frame discarded
            input_length.append(video.length) # Just use the video padded length to avoid CTC No path found error (v_len < a_len)
            source_str.append(align.sentence) # CHANGED [A] -> A, CHECK!

        source_str = np.array(source_str)
        label_length = np.array(label_length)
        input_length = np.array(input_length)
        Y_data = np.array(Y_data)
        X_data = np.array(X_data).astype(np.float32) / 255 # Normalize image data to [0,1], TODO: mean normalization over training data

        inputs = {'the_input': X_data,
                  'the_labels': Y_data,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function coz while training in forward ctc is zero

        return (inputs, outputs)

    @threadsafe_generator
    def next_train(self):
        r = np.random.RandomState(self.random_seed)
        while 1:
            # print "SI: {}, SE: {}".format(self.cur_train_index.value, self.shared_train_epoch.value)
            with self.cur_train_index.get_lock(), self.shared_train_epoch.get_lock():
                cur_train_index = self.cur_train_index.value
                self.cur_train_index.value += self.minibatch_size
                # Shared epoch increment on start or index >= training in epoch
                if cur_train_index >= self.steps_per_epoch * self.minibatch_size:
                    cur_train_index = 0
                    self.shared_train_epoch.value += 1
                    self.cur_train_index.value = self.minibatch_size
                if self.shared_train_epoch.value < 0:
                    self.shared_train_epoch.value += 1
                # Shared index overflow
                if self.cur_train_index.value >= self.training_size:
                    self.cur_train_index.value = self.cur_train_index.value % self.minibatch_size
                # Calculate differences between process and shared epoch
                epoch_differences = self.shared_train_epoch.value - self.process_train_epoch
            if epoch_differences > 0:
                self.process_train_epoch += epoch_differences
                for i in range(epoch_differences):
                    r.shuffle(self.train_list) # Catch up
                # print "GENERATOR EPOCH {}".format(self.process_train_epoch)
                # print self.train_list[0]
            # print "PI: {}, SI: {}, SE: {}".format(cur_train_index, self.cur_train_index.value, self.shared_train_epoch.value)
            # if self.curriculum is not None and self.curriculum.epoch != self.process_train_epoch:
            #     self.update_curriculum(self.process_train_epoch, train=True)
            # print "Train [{},{}] {}:{}".format(self.process_train_epoch, epoch_differences, cur_train_index,cur_train_index+self.minibatch_size)
            ret = self.get_batch(cur_train_index, self.minibatch_size, train=True)
            # if epoch_differences > 0:
            #     print "GENERATOR EPOCH {} - {}:{}".format(self.process_train_epoch, cur_train_index, cur_train_index + self.minibatch_size)
            #     print ret[0]['source_str']
            #     print "-------------------"
            yield ret

    @threadsafe_generator
    def next_val(self):
        while 1:
            with self.cur_val_index.get_lock():
                cur_val_index = self.cur_val_index.value
                self.cur_val_index.value += self.minibatch_size
                if self.cur_val_index.value >= self.validation_size:
                    self.cur_val_index.value = self.cur_val_index.value % self.minibatch_size
            # if self.curriculum is not None and self.curriculum.epoch != self.process_epoch:
            #     self.update_curriculum(self.process_epoch, train=False)
            # print "Val [{}] {}:{}".format(self.process_epoch, cur_val_index,cur_val_index+self.minibatch_size)
            ret = self.get_batch(cur_val_index, self.minibatch_size, train=False)
            yield ret

    def on_train_begin(self, logs={}):
        with self.cur_train_index.get_lock():
            self.cur_train_index.value = 0
        with self.cur_val_index.get_lock():
            self.cur_val_index.value = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.process_epoch = epoch

    # def update_curriculum(self, epoch, train=True):
    #     self.curriculum.update(epoch, train=train)
    #     print("Epoch {}: {}".format(epoch, self.curriculum))
