from keras import backend as K
import numpy as np
from spell import Spell
import sys, os
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_PATH)


# def _decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
#     """Decodes the output of a softmax.
#     Can use either greedy search (also known as best path)
#     or a constrained dictionary search.
#     # Arguments
#         y_pred: tensor `(samples, time_steps, num_categories)`
#             containing the prediction, or output of the softmax.
#         input_length: tensor `(samples, )` containing the sequence length for
#             each batch item in `y_pred`.
#         greedy: perform much faster best-path search if `true`.
#             This does not use a dictionary.
#         beam_width: if `greedy` is `false`: a beam search decoder will be used
#             with a beam of this width.
#         top_paths: if `greedy` is `false`,
#             how many of the most probable paths will be returned.
#     # Returns
#         Tuple:
#             List: if `greedy` is `true`, returns a list of one element that
#                 contains the decoded sequence.
#                 If `false`, returns the `top_paths` most probable
#                 decoded sequences.
#                 Important: blank labels are returned as `-1`.
#             Tensor `(top_paths, )` that contains
#                 the log probability of each decoded sequence.
#     """
#     decoded = K.ctc_decode(y_pred=y_pred, input_length=input_length,
#                            greedy=greedy, beam_width=beam_width, top_paths=top_paths)
#     paths = [path.eval(session=K.get_session()) for path in decoded[0]]
#     logprobs  = decoded[1].eval(session=K.get_session())

#     return (paths, logprobs)

def ascii_to_text(labels):
    # 26 is space, 27 is CTC blank char
    text = ''
    for c in labels:
        if c >= 0 and c < 26:
            text += chr(c + ord('a'))
        elif c == 26:
            text += ' '
    return text

PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'grid.txt')
spell = Spell(path=PREDICT_DICTIONARY)

# def decode(y_pred, input_length, greedy, beam_width, top_paths):
#     # language_model = kwargs.get('language_model', None)

#     # paths, logprobs = _decode(y_pred=y_pred, input_length=input_length, greedy=greedy, beam_width=beam_width, top_paths=top_paths)
#     decoded = K.ctc_decode(y_pred=y_pred, input_length=input_length, greedy=greedy, beam_width=beam_width, top_paths=top_paths)
#     paths = [path.eval(session=K.get_session()) for path in decoded[0]]
#     # logprobs  = decoded[1].eval(session=K.get_session())
#     # if language_model is not None:
#     #     # TODO: compute using language model
#     #     raise NotImplementedError("Language model search is not implemented yet")
#     # else:
#     #     # simply output highest probability sequence
#     #     # paths has been sorted from the start
#     result = paths[0]
#     return result

class Decoder(object):
    def __init__(self, beam_width):
        self.greedy         = False
        self.beam_width     = beam_width
        self.top_paths      = 1
        # self.language_model = kwargs.get('language_model', None)
        # self.postprocessors = postprocessors

    def decode(self, y_pred, input_length):
        # decoded = decode(y_pred, input_length, greedy=self.greedy, beam_width=self.beam_width, top_paths=self.top_paths)
        decoded = K.ctc_decode(y_pred=y_pred, input_length=input_length, greedy=self.greedy, beam_width=self.beam_width, top_paths=self.top_paths)
        paths = [path.eval(session=K.get_session()) for path in decoded[0]]
        decoded = paths[0]

        preprocessed = []
        for output in decoded:
            out = output
            # for postprocessor in self.postprocessors:
            #     out = postprocessor(out)
            out = ascii_to_text(out)
            out = spell.sentence(out)
            preprocessed.append(out)

        return preprocessed