from keras import backend as K
import numpy as np
from spell import Spell
import sys, os
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_PATH)


def ascii_to_text(labels):
    text = ''
    for c in labels:
        if c >= 0 and c < 26:
            text += chr(c + ord('a'))
        elif c == 26:
            text += ' '
    return text

PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'grid.txt')
spell = Spell(path=PREDICT_DICTIONARY)

class Decoder(object):
    def __init__(self, beam_width):
        self.greedy         = False
        self.beam_width     = beam_width
        self.top_paths      = 1

    def decode(self, y_pred, input_length):
        decoded = K.ctc_decode(y_pred=y_pred, input_length=input_length, greedy=self.greedy, beam_width=self.beam_width, top_paths=self.top_paths)
        paths = [path.eval(session=K.get_session()) for path in decoded[0]]
        decoded = paths[0]

        preprocessed = []
        for output in decoded:
            out = output
            out = ascii_to_text(out)
            out = spell.sentence(out)
            preprocessed.append(out)

        return preprocessed