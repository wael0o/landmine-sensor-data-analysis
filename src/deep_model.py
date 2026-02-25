import tensorflow as tf
from tensorflow.keras import layers, models


def build_dense_classifier(input_dim, num_classes):

    model = models.Sequential([
        layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        layers.BatchNormalization(),

        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),

        layers.Dense(16, activation="relu"),

        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model