import numpy as np

lines = None

def text_to_ascii(text):
    ret = []
    for char in text:
        if char >= 'a' and char <= 'z':
            ret.append(ord(char) - ord('a'))
        elif char == ' ':
            ret.append(26)
    return ret

class Align(object):
    def __init__(self, absolute_max_string_len=32):
        self.absolute_max_string_len = absolute_max_string_len

    def from_file(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        align = [(int(y[0])/1000, int(y[1])/1000, y[2]) for y in [x.strip().split(" ") for x in lines]]
        self.build(align)
        return self

    def from_array(self, align):
        self.build(align)
        return self

    def build(self, align):
        rem = ['sp','sil'] # Remove these
        # self.align = [sub for sub in align if sub[2] not in rem]
        self.sentence = " ".join([y[-1] for y in align ])
        self.sentence = self.sentence.replace('sp','')
        self.sentence = self.sentence.replace('sil','')
        self.sentence = self.sentence.strip()
        print("Sentence ",self.sentence)
        self.label = text_to_ascii(self.sentence)
        self.padded_label = self.get_padded_label(self.label)

    def get_padded_label(self, label):
        print("Absolute max string length ",self.absolute_max_string_len)
        print("length label ",len(label))
        padding = np.ones((self.absolute_max_string_len-len(label))) * -1
        return np.concatenate((np.array(label), padding), axis=0)

    @property
    def word_length(self):
        return len(self.sentence.split(" "))

    @property
    def sentence_length(self):
        return len(self.sentence)

    @property
    def label_length(self):
        return len(self.label)
