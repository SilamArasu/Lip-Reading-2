import numpy as np
# from videos import VideoAugmenter
from videos import Video


class KeyFrame(object):

    def extract(self, video):
        original_video_length = video.length
        video = self.remove_redundant(video)
        video_unpadded_length = video.length
        if video.length != original_video_length:
          video = self.pad(video, original_video_length)
        return video

    def remove_redundant(self, video):
        first = video.mouth[0]
        # diff_dict = dict()
        threshold = 300 
        res = None
        org_index = 0
        new_mouth = []
        new_mouth.append(first)
        for index in range(len(video.mouth)-1):
            second = video.mouth[index+1]
            filename = index + 1
            res = self.find_diff(first, second)
            diff = np.count_nonzero(res)
            if diff > threshold:
                new_mouth.append(second)
            org_index += 1
            # diff_dict[filename+1] = diff
            first = second
        new_mouth = np.array(new_mouth)

        new_video = Video(video.vtype, video.face_predictor_path)
        new_video.mouth = new_mouth
        new_video.set_data(new_video.mouth)
        return new_video    


    def tobw(self,img):
        img = img.dot([0.07, 0.72, 0.21])
        i = 0
        while i <= 256:
            img[(img>i-16)*(img<i)] = i
            i += 16
        return img

    def find_diff(self,img1,img2):
        img1 = self.tobw(img1)
        img2 = self.tobw(img2)
        subt = np.fabs(np.subtract(img1,img2))      
        return subt
        
    def pad(self, video, length):
        pad_length = max(length - video.length, 0)
        video_length = min(length, video.length)
        mouth_padding = np.zeros((pad_length, video.mouth.shape[1], video.mouth.shape[2], video.mouth.shape[3]), dtype=np.uint8)
        new_video = Video(video.vtype, video.face_predictor_path)
        new_video.mouth = np.concatenate((video.mouth[0:video_length], mouth_padding), 0)
        new_video.set_data(new_video.mouth)
        return new_video    
    