from nltk.grammar import PCFG
from nltk.parse.pchart import LongestChartParser
from nltk.parse.generate import generate
import numpy as np

def print_if_true(text, flag):
    if flag:
        print(text)

class LCModel:
    def __init__(self, grammar: PCFG, deletion_rate: float, max_depth: int = None):
        parser = LongestChartParser(grammar)

        # generate all possible sequences from the grammar
        self.language: list[tuple[list[str], np.float64]] = []
        for sequence in generate(grammar, depth = max_depth):
            sequence_prob = next(parser.parse(sequence)).prob()
            self.language.append((sequence, np.float64(sequence_prob)))

        # add subsequences to the language
        sub_sequences = []
        sub_sequence_probs = []
        for (language_sequence, _) in self.language:
            for i in range(1, len(language_sequence)):
                sub_sequence = language_sequence[:i]
                if sub_sequence in sub_sequences:
                    continue

                sub_sequence_prob = np.float64(0.0)
                for (sequence, prob) in self.language:
                    if sequence[:i] == sub_sequence:
                        sub_sequence_prob += prob

                sub_sequences.append(sub_sequence)
                sub_sequence_probs.append(sub_sequence_prob)

        self.language += list(zip(sub_sequences, sub_sequence_probs))

        self.deletion_rate = np.float64(deletion_rate)


    def get_prob(self, sequence: list[str]) -> np.float64:
        """Calculate the a priori probability of `sequence` [p_L(sequence)]."""
        for (language_sequence, probability) in self.language:
            if language_sequence == sequence:
                return probability
            
        return np.float64(0.0)


    def get_distortions(self, sequence: list[str]) -> list[tuple[list[str], np.float64]]:
        """
        Generate all possible memory representations/distortions from a given sequence.

        Args
        ----
        sequence : list[str]
            A sequence of words from the grammar.

        Returns
        -------
        list[tuple[list[str], np.float64]]
            A list of tuples with the form (distortion, distortion_probability)
        """
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
        """
        Find all language sequences which could have given rise to the given memory
        representation/distortion.

        Args
        ----
        distortion : list[str]
            A sequence of words from the grammar representing a
            distorted context.

        Returns
        -------
        list[list[str]]
            All language sequences which contain all of the words in
            `distortion`. 
        """
        reconstructions = []
        for (reconstruction, _) in self.language:
            if all([word in reconstruction for word in distortion]):
                reconstructions.append(reconstruction)

        return reconstructions


    def calculate_processing_difficulty(self, sequence: list[str], verbose = False) -> np.float64:
        """
        Calculate the predicted processing difficulty of the last word in `sequence`.

        See the thesis for an explanation of lossy-context surprisal and details on this implementation

        Args
        ----
        sequence : list[str]
            A sequence of words from the grammar, with the last being the word for which
            processing difficulty is calculated.

        Returns
        -------
        np.float64
            The processing difficulty.
        """
        print_if_true(f"True context: {' '.join(sequence)}", flag = verbose)
        target_word = sequence[-1]
        processing_difficulty = np.float64(0.0)
        for (distortion, probability) in self.get_distortions(sequence[:-1]):
            print_if_true(f"Current distortion: {distortion}", flag = verbose)
            print_if_true(f"p(r|c) = {probability}", flag = verbose)
            average_prob = np.float64(0.0)
            distortion_prob_sum = np.float64(0.0)
            for reconstruction in self.get_reconstructions(distortion):
                reconstruction_with_target = reconstruction + [target_word]
                context_probability = self.get_prob(reconstruction_with_target)
                if not context_probability > 0:
                    continue

                print_if_true(f" ## Possible reconstructed context: {' '.join(reconstruction)}", flag = verbose)

                print_if_true(f" ## Reconstructing sentence as: {' '.join(reconstruction_with_target)}", flag = verbose)
                distortion_probability = self.deletion_rate**(len(reconstruction) - len(distortion)) * \
                    (1-self.deletion_rate)**len(distortion)
                print_if_true(f" ## p(r|~c) = {distortion_probability}", flag = verbose)

                print_if_true(f" ## p(w_1,...,w_[i-1],w_i) = {context_probability}\n", flag = verbose)

                average_prob += context_probability * distortion_probability
                distortion_prob_sum += distortion_probability

            average_prob /= distortion_prob_sum

            if not average_prob > 0:
                continue

            print_if_true(f"E[p(w|~c)] = {average_prob}", verbose)

            processing_difficulty += -np.log(average_prob) * probability
            print()

        return processing_difficulty


if __name__ == "__main__":
    import grammars

    del_rate = 0.1

    grammar = grammars.gen_russian_grammar_exp1(0.7, 0.8, 0.6)

    # grammar = PCFG.fromstring(grammar)
    # model = LCModel(grammar, del_rate)
    grammar = PCFG.fromstring(grammars.gen_russian_grammar_exp2(0.8, 0.8, 0.55, 0.9, 0.9))
    model = LCModel(grammar, del_rate)

    orc_non_local = model.calculate_processing_difficulty("RPAcc Subj V".split(), False) # This should be greater...
    # print(orc_non_local)
    orc_local = model.calculate_processing_difficulty("RPAcc V".split(), False) # ...than this
    # print(orc_local)

    src_local = model.calculate_processing_difficulty("RPNom V".split()) # This should be smaller...
    src_non_local = model.calculate_processing_difficulty("RPNom DO V".split(), False) # ...than this

    print(orc_non_local, orc_local)
    print(src_non_local, src_local)

    orc_diff = orc_local - orc_non_local
    src_diff = src_local - src_non_local

    print(orc_diff)
    print(src_diff)

    # # Russian experiment 2
    no_intv = model.calculate_processing_difficulty("RPNom V".split())

    # # one_adj = model.calculate_processing_difficulty("RPNom PP V".split())
    # # two_adj = model.calculate_processing_difficulty("RPNom PP PP V".split())

    one_arg = model.calculate_processing_difficulty("RPNom DO V".split())
    two_arg = model.calculate_processing_difficulty("RPNom DO IO V".split())

    print(no_intv)
    # # print(one_adj)
    # # print(two_adj)
    print(one_arg)
    print(two_arg)