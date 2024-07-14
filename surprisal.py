from nltk.grammar import PCFG
from nltk.parse.pchart import LongestChartParser
from nltk.parse.generate import generate
import numpy as np

class SurprisalModel:
    def __init__(self, grammar: PCFG, max_depth = None):
        parser = LongestChartParser(grammar, depth = max_depth)

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

    def calculate_processing_difficulty(self, sequence: list[str]) -> np.float64 | None:
        """
        Calculates the probability of the last word in the sequence given the preceeding words.

        Args
        ----
        sequence : list[str]
            A sequence of words from the grammar with the last being the
            target word.
        
        Returns
        np
        """
        context_prob = np.float64(0.0)
        sentence_prob = np.float64(0.0)
        context = sequence[:-1]
        for (language_sequence, probability) in self.language:
            if language_sequence == sequence:
                sentence_prob = probability
            elif language_sequence == context:
                context_prob = probability

            if context_prob > 0 and sentence_prob > 0:
                break

        if context_prob == 0:
            return None

        return sentence_prob/context_prob