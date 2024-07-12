from nltk.grammar import PCFG
from nltk.parse.pchart import LongestChartParser
from nltk.parse.generate import generate
import numpy as np

class Model:
    def __init__(self, grammar: PCFG, deletion_rate: float):
        self.grammar = grammar
        self.parser = LongestChartParser(self.grammar)
        self.deletion_rate = np.float64(deletion_rate)

    def get_prob(self, sequence: list[str]) -> np.float64:
        trees = list(self.parser.parse(sequence))
        if len(trees) > 0:
            return np.float64(trees[0].prob())
        else:
            return np.float64(0.0)
    
    def get_distortions(self, sequence: list[str]) -> list[tuple[list[str], np.float64]]:
        distortions = []
        # length is the length of the distorted sequence
        for length in range(len(sequence), -1, -1):
            prob = self.deletion_rate**(len(sequence) - length) * (1-self.deletion_rate)**length
            distortions += [(distortion, prob)
                            for distortion in self._get_distortions_of_length(sequence, length)]
            
        return distortions

    def _get_distortions_of_length(self, sequence: list[str], length: int) -> list[list[str]]:
        if length == len(sequence):
            return [sequence]
        elif length == 0:
            return [[]]

        distortions = []
        for (i, word) in enumerate(sequence):
            if length == 1:
                distortions.append([word])
            else:
                distortions += [[word] + distortion for distortion in self._get_distortions_of_length(sequence[i+1:], length - 1)]


        return distortions
    
    def get_reconstructions(self, distortion: list[str]) -> list[list[str]]:
        reconstructions = []
        for reconstruction in self.language:
            if all([word in reconstruction for word in distortion]):
                reconstructions.append(reconstruction)

        return reconstructions
    
    def calculate_processing_difficulty(self, sequence: list[str]) -> np.float64:
        target_word = sequence[-1]
        processing_difficulty = np.float64(0.0)
        for (distortion, probability) in self.get_distortions(sequence[:-1]):
            # print(f"Current distortion: {distortion}")
            # print(f"p = {probability}")
            # print(self.get_reconstructions(distortion))
            average_prob = np.float64(0.0)
            for reconstruction in self.get_reconstructions(distortion):
                reconstruction_with_target = reconstruction[:-1] + [target_word]
                # print(f"Reconstructing as {' '.join(reconstruction_with_target)}")
                distortion_probability = self.deletion_rate**(len(reconstruction) - (len(distortion) + 1)) * \
                    (1-self.deletion_rate)**len(distortion)
                # print(f"dist p = {distortion_probability}")
                
                context_probability = self.get_prob(reconstruction_with_target)

                # print(f"context p = {context_probability}")

                average_prob += context_probability * distortion_probability

            if not average_prob > 0:
                continue

            processing_difficulty += -np.log(average_prob) * probability

        return processing_difficulty


if __name__ == "__main__":
    import grammars

    del_rate = 0.03

    grammar = grammars.gen_russian_grammar_exp1(0.7, 0.8, 0.6)

    # grammar = PCFG.fromstring(grammar)
    # model = Model(grammar, del_rate)
    grammar = PCFG.fromstring(grammars.gen_russian_grammar_exp2(0.8, 0.8, 0.6, 0.9, 0.9))
    model = Model(grammar, del_rate)

    orc_canon = model.calculate_processing_difficulty("RPAcc Subj V".split()) # This should be smaller...
    orc_non_canon = model.calculate_processing_difficulty("RPAcc V".split()) # ...than this

    src_canon = model.calculate_processing_difficulty("RPNom V".split()) # This should be smaller...
    src_non_canon = model.calculate_processing_difficulty("RPNom DO V".split()) # ...than this

    print(orc_canon, orc_non_canon)
    print(src_canon, src_non_canon)

    orc_diff = orc_canon - orc_non_canon
    src_diff = src_canon - src_non_canon

    print(orc_diff)
    print(src_diff)

    # Russian experiment 2
    grammar = PCFG.fromstring(grammars.gen_russian_grammar_exp2(0.8, 0.8, 0.6, 0.8, 0.8))
    model = Model(grammar, del_rate)

    no_intv = model.calculate_processing_difficulty("RPNom V".split())

    # one_adj = model.calculate_processing_difficulty("RPNom PP V".split())
    # two_adj = model.calculate_processing_difficulty("RPNom PP PP V".split())

    one_arg = model.calculate_processing_difficulty("RPNom DO V".split())
    two_arg = model.calculate_processing_difficulty("RPNom DO IO V".split())

    print(no_intv)
    # print(one_adj)
    # print(two_adj)
    print(one_arg)
    print(two_arg)