import re
import string
from collections import Counter

path = '/home/arasu/FYP/lipreading_code/Training/grid.txt'

class Spell(object):
    def __init__(self, path):
        self.dictionary = Counter(self.words(open(path).read()))

    def words(self, text):
        return re.findall(r'\w+', text.lower())

    def P(self, word, N=None):
        "Probability of `word`."
        N = sum(self.dictionary.values())
        return self.dictionary[word] / N

    def correction(self, word):
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.P)

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        known = self.known([word])
        if len(known) != 0:
                return known
        ed1 = self.known(self.edits1(word))
        if len(ed1) != 0:
                return ed1
        ed2 = self.known((e2 for e1 in self.edits1(word) for e2 in self.edits1(e1)))
        if len(ed2) != 0:
                return ed2
        return known

    def known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.dictionary)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    # Correct words
    def corrections(self, words):
        return [self.correction(word) for word in words]

    # Correct sentence
    def sentence(self, sentence):
        return ' '.join(self.corrections(sentence.split()))