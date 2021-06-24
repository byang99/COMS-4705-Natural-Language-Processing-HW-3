"""

COMS 4705 Natural Language Processing
Name: Brian Yang
UNI: by2289

6/1/2021

"""

from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from extract_training_data import FeatureExtractor, State
import sys
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


class Parser(object):

    def __init__(self, extractor, modelfile):
        self.model = tf.keras.models.load_model(modelfile)
        self.extractor = extractor

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1, len(words)))
        state.stack.append(0)

        # While buffer is not empty
        while state.buffer:
            # TODO: Write the body of this loop for part 4

            # Use feature extractor to obtain a representation of the current
            features = self.extractor.get_input_representation(words, pos, state)

            # Next, use our model to retrieve a softmax activated vector of
            # possible actions
            prob_vec = self.model.predict(np.expand_dims(features, axis=0))
            prob_vec = prob_vec.flatten()

            # Select the highest-scoring permitted transition.
            # Create a list of possible actions
            list_of_possible_actions = list()
            for i in range(len(self.output_labels)):
                action, relation = self.output_labels[i]
                prob = prob_vec[i]
                list_of_possible_actions.append((prob, action, relation))


            # Sort these possible actions according to output probability
            list_of_possible_actions.sort(key=lambda x: x[0], reverse=True)

            # Go through the list until we find a legal transition
            for action_tup in list_of_possible_actions:
                action = action_tup[1]
                dep_relation = action_tup[2]

                # If stack is empty, arc-left and arc-right are not permitted
                if action == "left_arc":
                    # If stack is empty, arc-left is not permitted
                    if len(state.stack) == 0 or len(state.buffer) == 0:
                        continue

                    # The root node must never be the target of an arc-left
                    if len(state.stack) == 1:
                        continue

                    # arc-left is valid - update state
                    state.left_arc(dep_relation)
                    break

                elif action == "right_arc":
                    # If stack is empty, arc-left and arc-right are not permitted
                    if len(state.stack) == 0 or len(state.buffer) == 0:
                        continue

                    # arc_right is valid - update state
                    state.right_arc(dep_relation)
                    break

                elif action == "shift":
                    # If buffer size is 1, we cannot shift out of the buffer, unless the
                    # stack is empty
                    if len(state.buffer) == 1 and len(state.stack) != 0:
                        continue

                    if len(state.buffer) == 0:
                        continue

                    # shift is valid - update state
                    state.shift()
                    break

        result = DependencyStructure()
        for p, c, r in state.deps:
            result.add_deprel(DependencyEdge(c, words[c], pos[c], p, r))
        return result


if __name__ == "__main__":

    # For PyCharm
    # WORD_VOCAB_FILE = 'data/words.vocab'
    # POS_VOCAB_FILE = 'data/pos.vocab'

    # For Google Colab
    WORD_VOCAB_FILE = '/content/hw3_files/data/words.vocab'
    POS_VOCAB_FILE = '/content/hw3_files/data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE, 'r')
        pos_vocab_f = open(POS_VOCAB_FILE, 'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2], 'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
