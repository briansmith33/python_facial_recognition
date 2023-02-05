from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow import keras
from threading import Thread
import tensorflow as tf
import numpy as np
import time
import cv2


class Recognize(Thread):
    def __init__(self, data_dir="./faces", model_name="recognition.h5", interval=5):
        super().__init__()
        self.data_dir = data_dir
        self.model_name = model_name
        self.interval = interval
        self.batch_size = 32
        self.img_height = 180
        self.img_width = 180
        self.train_ds = None
        self.val_ds = None
        self.class_names = None
        self.init()

    def init(self):
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.class_names = self.train_ds.class_names

        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    def create_model(self):
        num_classes = len(self.class_names)

        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal", input_shape=(self.img_height, self.img_width, 3)),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
            ]
        )

        model = Sequential([
            data_augmentation,
            layers.Rescaling(1. / 255),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, name="outputs")
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.fit(self.train_ds, validation_data=self.val_ds, epochs=15)

        model.save(self.model_name)

    def run(self):
        model = load_model(self.model_name)
        video_capture = cv2.VideoCapture(0)

        counter = 0
        while True:
            ret, frame = video_capture.read()
            if ret:
                small_frame = cv2.resize(frame, (self.img_width, self.img_height))

                rgb_small_frame = small_frame[:, :, ::-1]

                if counter == 0:
                    pil_img = tf.keras.utils.array_to_img(rgb_small_frame)
                    img_array = tf.keras.utils.img_to_array(pil_img)
                    img_array = tf.expand_dims(img_array, 0)  # Create a batch

                    predictions = model.predict(img_array)
                    score = tf.nn.softmax(predictions[0])

                    print("{} {:.2f}".format(self.class_names[np.argmax(score)], 100 * np.max(score)))

                if counter == self.interval:
                    counter = 0
                else:
                    counter += 1

                time.sleep(1)

