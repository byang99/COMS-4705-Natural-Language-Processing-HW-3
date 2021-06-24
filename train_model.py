"""

COMS 4705 Natural Language Processing
Name: Brian Yang
UNI: by2289

6/1/2021

"""

from extract_training_data import FeatureExtractor
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Embedding, Dense

tf.compat.v1.disable_eager_execution()


def build_model(word_types, pos_types, outputs):
    # TODO: Write this function for part 3
    model = Sequential()

    model.add(Embedding(input_dim=word_types, output_dim=32, input_length=6))
    model.add(Flatten())
    model.add(Dense(units=100, activation="relu"))
    model.add(Dense(units=10, activation="relu"))
    model.add(Dense(outputs, activation=keras.activations.softmax))

    model.compile(keras.optimizers.Adam(learning_rate=0.01), loss="categorical_crossentropy")
    return model


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
    print("Compiling model.")
    model = build_model(len(extractor.word_vocab), len(extractor.pos_vocab), len(extractor.output_labels))
    inputs = np.load(sys.argv[1])
    outputs = np.load(sys.argv[2])
    print("Done loading data.")

    # Now train the model
    model.fit(inputs, outputs, epochs=5, batch_size=100)

    model.save(sys.argv[3])
