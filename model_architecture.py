"""
MITRA: Deep Learning-Based Emotion Recognition Model
Author: Lokesh Manepalli
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout


def build_mitra_model(vocab_size, max_length, embedding_dim=128, num_classes=6):
    """
    Builds the MITRA BiLSTM-CNN hybrid architecture.
    """

    inputs = Input(shape=(max_length,))

    # Embedding Layer
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)

    # BiLSTM Layer (context understanding)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    # CNN Layer (local feature extraction)
    x = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)

    # Dense Layer
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    return model
