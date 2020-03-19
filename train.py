import sys, os
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_PATH)

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from keras.optimizers import Adam
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from generators import Generator
from callbacks import Metrics
from model import Network
import numpy as np
import datetime

np.random.seed(34)

DATASET_DIR  = os.path.join(CURRENT_PATH, 'datasets')
OUTPUT_DIR   = os.path.join(CURRENT_PATH, 'results')
LOG_DIR      = os.path.join(CURRENT_PATH, 'logs')

def train(run_name, start_epoch, stop_epoch, img_c, img_w, img_h, frames_n, absolute_max_string_len, output_size,minibatch_size):
    
    gen = Generator(dataset_path=DATASET_DIR,
                                minibatch_size=minibatch_size,
                                img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                                absolute_max_string_len=absolute_max_string_len)

    net = Network(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                            absolute_max_string_len=absolute_max_string_len, output_size=output_size)
    net.summary()

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    net.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)

    # load preexisting trained weights for the model
    if start_epoch > 0:
        weight_file = os.path.join(OUTPUT_DIR, os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
        net.model.load_weights(weight_file)

    try:
        os.makedirs(os.path.join(LOG_DIR, run_name), exist_ok=True)
    except:
        pass

    metrics  = Metrics(net, gen.next_val(), 100, minibatch_size, os.path.join(OUTPUT_DIR, run_name))
    csv_logger  = CSVLogger(os.path.join(LOG_DIR, "{}-{}.csv".format('training',run_name)), separator=',', append=True)
    checkpoint  = ModelCheckpoint(os.path.join(OUTPUT_DIR, run_name, "weights{epoch:02d}.h5"), monitor='val_loss', save_weights_only=True, mode='auto', period=1)

    net.model.fit_generator(generator=gen.next_train(),
                        steps_per_epoch=gen.default_training_steps, epochs=stop_epoch,
                        validation_data=gen.next_val(), validation_steps=gen.default_validation_steps,
                        callbacks=[checkpoint, metrics, gen, csv_logger],
                        initial_epoch=start_epoch,
                        verbose=1,
                        max_q_size=5,
                        workers=2,
                        pickle_safe=True)

if __name__ == '__main__':
    run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    # for testing in laptop
    # train(run_name, 0, 2, 3, 100, 50, 75, 32,28, 2)
    # for real training
    train(run_name, 0, 10, 3, 100, 50, 75, 32,28, 50)
    # absolute_max_string_len = 32 coz the max length of sentence spoken(in align) is 31
    # output_size = 28 coz average len sentence 24. Rounding to 28
    print("Training finished")
