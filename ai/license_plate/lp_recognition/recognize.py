import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, glob, re, time, sys
from os.path import splitext,basename
import tensorflow as tf
from keras.models import model_from_json
from tensorflow import keras
import tensorflow as tf
from PIL import Image

max_length = 8
characters = ['*', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M',
              'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']

def img_process(img, target_h, target_w):
    im_shape = img.shape
    im_size_h = im_shape[0]
    im_size_w = np.max(im_shape[0:2])
    im_scale = float(target_h) / float(im_size_h)
    if np.round(im_scale * im_size_w) > target_w:
        im_scale = float(target_w) / float(im_size_w)
    image = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    new_h, new_w = image.shape
    delta_h = max(0, target_h - new_h)
    delta_w = max(0,  target_w - new_w )
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255,255,255])
#     image[image > 230] = 255
    return image

def decode_batch_predictions(pred):
    char_to_num = tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=sorted(list(characters)), num_oov_indices=0, mask_token=None
    )

    # Mapping integers back to original characters
    num_to_char = tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )

    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

def build_model(img_height,img_width):
    # Inputs to the model
    input_img = tf.keras.layers.Input(
        shape=(img_height,img_width, 1), name="image", dtype="float32"
    )
    labels = tf.keras.layers.Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = tf.keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # RNNs
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = tf.keras.layers.Dense(len(characters) + 1, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = tf.keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    opt = tf.keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    return model

class LP_Recognize:
    __shared_instance = None
    @staticmethod
    def getInstance(): 
        """Static Access Method"""
        if LP_Recognize.__shared_instance == None: 
            LP_Recognize() 
        return LP_Recognize.__shared_instance
    def __init__(self):
        self.img_height = 64
        self.img_width = 256
        self.max_length = 8
        if LP_Recognize.__shared_instance != None: 
            raise Exception ("This class is a singleton class !") 
        else: 
            # Singleton Pattern Design only instantiate the model once
            self.model = build_model(img_height=self.img_height,img_width=self.img_width)
            self.model.load_weights('ai/models/CV_together_Weight.h5')
            self.model = keras.models.Model(
                self.model.get_layer(name="image").input, self.model.get_layer(name="dense2").output
            )     
            LP_Recognize.__shared_instance = self

    def rec(self, image):
        # image = cv2.imread(image_path, 0)
        image = (255*image[0]).astype(np.uint8)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        crop = img_process(image, self.img_height, self.img_width)
        crop = np.expand_dims(crop.astype('float32') / 255, axis = -1)
        crop = np.expand_dims(crop, axis = 0)
        preds = self.model.predict(crop)
        t = decode_batch_predictions(preds)

        return t[0].replace("*", "")


