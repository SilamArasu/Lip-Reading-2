import numpy as np
# from videos import VideoAugmenter
from videos import Video

class KeyFrame():
    def __init__(self):
        self.sentence_length = -1
        self.flip_probability = 0.5
        self.jitter_probability = 0.05

    # def update(self, epoch, train=True):
    #     self.epoch = epoch
    #     self.train = train
    #     current_rule = self.rules(self.epoch)
    #     self.sentence_length = current_rule.get('sentence_length') or -1
    #     self.flip_probability = current_rule.get('flip_probability') or 0.0
    #     self.jitter_probability = current_rule.get('jitter_probability') or 0.0

    def apply(self, video, align):
        original_video_length = video.length
        # if train:
        if np.random.ranf() < self.flip_probability:
            video = self.horizontal_flip(video)
        video = self.temporal_jitter(video)
        video_unpadded_length = video.length
        if video.length != original_video_length:
          video = self.pad(video, original_video_length)
        return video, align

    def horizontal_flip(self, video):
        new_video = Video(video.vtype, video.face_predictor_path)
        new_video.face = np.flip(video.face, 2)
        new_video.mouth = np.flip(video.mouth, 2)
        new_video.set_data(new_video.mouth)
        return new_video

    def temporal_jitter(self, video):
        changes = [] # [(frame_i, type=del/dup)]
        t = video.length
        for i in range(t):
            if np.random.ranf() <= self.jitter_probability/2:
                changes.append((i, 'del'))
            if self.jitter_probability/2 < np.random.ranf() <= self.jitter_probability:
                changes.append((i, 'dup'))
        new_face = np.copy(video.face)
        new_mouth = np.copy(video.mouth)
        pos = 0
        for change in changes:
            actual_pos = change[0] + pos
            if change[1] == 'dup':
                new_face = np.insert(new_face, actual_pos, new_face[actual_pos], 0)
                new_mouth = np.insert(new_mouth, actual_pos, new_mouth[actual_pos], 0)
                pos = pos + 1
            else:
                new_face = np.delete(new_face, actual_pos, 0)
                new_mouth = np.delete(new_mouth, actual_pos, 0)
                pos = pos - 1
        new_video = Video(video.vtype, video.face_predictor_path)
        new_video.face = new_face
        new_video.mouth = new_mouth
        new_video.set_data(new_video.mouth)
        return new_video    

    def pad(self, video, length):
        pad_length = max(length - video.length, 0)
        # if pad_length == 0:
        #     # print("--------------")
        #     # print("Here pad = 0, video type ",type(video))
        #     return video
        # else:
        #     print("--------------")
        #     print("Here pad = 0, video type ",type(video))
        video_length = min(length, video.length)
        face_padding = np.zeros((pad_length, video.face.shape[1], video.face.shape[2], video.face.shape[3]), dtype=np.uint8)
        mouth_padding = np.zeros((pad_length, video.mouth.shape[1], video.mouth.shape[2], video.mouth.shape[3]), dtype=np.uint8)
        new_video = Video(video.vtype, video.face_predictor_path)
        new_video.face = np.concatenate((video.face[0:video_length], face_padding), 0)
        new_video.mouth = np.concatenate((video.mouth[0:video_length], mouth_padding), 0)
        new_video.set_data(new_video.mouth)
        return new_video    

    def __str__(self):
        return "{}(train: {}, sentence_length: {}, flip_probability: {}, jitter_probability: {})"\
            .format(self.__class__.__name__, self.train, self.sentence_length, self.flip_probability, self.jitter_probability)
    