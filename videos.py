import sys, os
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_PATH)

import numpy as np
from scipy import ndimage
from scipy.misc import imresize
import skvideo.io
import dlib
from aligns import Align

class Video(object):
    def __init__(self, face_predictor_path=None):
        self.face_predictor_path = face_predictor_path

    def from_frames(self, path):
        frames_path = sorted([os.path.join(path, x) for x in os.listdir(path)])
        frames = [ndimage.imread(frame_path) for frame_path in frames_path]
        self.process_frames_mouth(frames)
        return self

    def process_frames_mouth(self, frames):
        self.mouth = np.array(frames)
        self.set_data(frames)

    def from_video(self, path):
        frames = self.get_video_frames(path)
        self.process_frames_face(frames)
        return self

    def process_frames_face(self, frames):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.face_predictor_path)
        mouth_frames = self.get_frames_mouth(detector, predictor, frames)
        self.mouth = np.array(mouth_frames)
        self.set_data(mouth_frames)



    def get_frames_mouth(self, detector, predictor, frames):
        MOUTH_WIDTH = 100
        MOUTH_HEIGHT = 50
        HORIZONTAL_PAD = 0.19
        normalize_ratio = None
        mouth_frames = []
        for frame in frames:
            dets = detector(frame, 1)
            shape = None
            for k, d in enumerate(dets):
                shape = predictor(frame, d)
                i = -1
            if shape is None: # Detector doesn't detect face, just return as is
                return frames
            mouth_points = []
            for part in shape.parts():
                i += 1
                if i < 48: # Only take mouth region
                    continue
                mouth_points.append((part.x,part.y))
            np_mouth_points = np.array(mouth_points)

            mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

            if normalize_ratio is None:
                mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
                mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

                normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

            new_img_shape = (int(frame.shape[0] * normalize_ratio), int(frame.shape[1] * normalize_ratio))
            resized_img = imresize(frame, new_img_shape)

            mouth_centroid_norm = mouth_centroid * normalize_ratio

            mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
            mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH / 2)
            mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
            mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT / 2)

            mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]

            mouth_frames.append(mouth_crop_image)
        return mouth_frames

    def get_video_frames(self, path):
        videogen = skvideo.io.vreader(path)
        frames = np.array([frame for frame in videogen])
        return frames

    def set_data(self, frames):
        data_frames = []
        for frame in frames:
            frame = frame.swapaxes(0,1) # swap width and height to form format W x H x C

            data_frames.append(frame)
        frames_n = len(data_frames)
        data_frames = np.array(data_frames) # T x W x H x C
        self.data = data_frames
        self.length = frames_n
