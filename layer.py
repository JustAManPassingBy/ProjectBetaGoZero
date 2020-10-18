from keras.layers import Conv2D, Dense, Flatten, Input, BatchNormalization, Activation, MaxPooling2D, ZeroPadding2D, Add
from keras.models import Model
from keras.models import load_model
from keras import regularizers
#from keras import models, layers
#from keras import Input
#from keras.models import Model, load_model
#from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
#from keras.callbacks import ModelCheckpoint, EarlyStopping

from game import GameState

from math import *
import numpy as np
from matplotlib import pyplot as plt

from definitions import PASS_MOVE, BLACK, EMPTY, WHITE, KOMI
from definitions import MAX_VALUE, MIN_VALUE
from definitions import BOARD_SIZE, SQUARE_BOARD_SIZE
from definitions import PARTIAL_OBSERVABILITY, ADDITIONAL_SEARCH_COUNT_DURING_MCTS
from definitions import SAVE_PERIOD

#  -- Model --
#  - if Board size is 19
#  1. Input           : 19 x 19 x PARTIAL_OBSERVABILITY
#
#  2. Zero Padding 1  : 2D Padding, Additional (3 x 3) Layer
#  3. Conv 1          : 19 x 19 x 16 , 16 filters, 3 x 3 filter size, stride (1, 1)
#  4. Batch Norm 1
#  5. Relu activation 1
#  6. Max pooling 1   : 10 x 10 x 16 , 2 x 2 pool size , strides (2, 2)
#
#  7. Conv 2          : 10 x 10 x 32 , 32 filters, 1 x 1 filter size, strides (1, 1)
#  8. Batch Norm 2
#  9. Relu activation 2
#
# 10. Conv 3          : 10 x 10 x 64 , 64 filters, 2 x 2 filter size, strides (1, 1)
# 11. Batch Norm 3
# 12. Relu activation 3
#
# 13 ~ 15 : Shortcut(From [Max Pooling 1])
# 13. Conv 4          : 10 x 10 x 64 , 64 filters, 3 x 3 filter size, strides (1, 1)
# 14. Bacth Norm 4
# 15. Relu activation 4
#
# 16. Add             : 12. Relu activation 3 and 15. Relu activation 4
# 17. Max Pooling 2   : 5 x 5 x 64 , 2 x 2 pool size , strides (2, 2)
# 18. Flatten 64 x 5 x 5 -> 1600
# 19. Dense 1         : Relu activation, 1024
# 20. Output          : Spot - [0 : (19 x 19 + 1)] - 0 ~ 1 (Sigmoid) / WinProb - [-1 : 1] - -1 ~ 1(Tanh)


#  5. Conv 4  : 10 x 10 x 64 , 64 filters, 3 x 3 , stride (1, 2)
#  5. Flatten
#  6. Dense 1 : 4096, Relu
#  6. Dense 2 : 1024, Relu
#  7. Output : (Spot -  [0 : 19 x 19] - 0 ~ 1[Sigmoid]) / (WinProb - [0 : 1] - 0 ~ 1[Sigmoid])
def create_keras_layer():
        input_buffer = Input(shape=(BOARD_SIZE, BOARD_SIZE, PARTIAL_OBSERVABILITY))

        zero_padding_1 = ZeroPadding2D(padding=(3, 3))(input_buffer)
        conv_1 = Conv2D(16, (3, 3), padding='same', strides=(1, 1))(zero_padding_1)
        batch_norm_1 = BatchNormalization()(conv_1)
        relu_1 = Activation('relu')(batch_norm_1)
        max_pool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(relu_1)

        conv_2 = Conv2D(32, (1, 1), padding='same', strides=(1, 1))(max_pool_1)
        batch_norm_2 = BatchNormalization()(conv_2)
        relu_2 = Activation('relu')(batch_norm_2)

        conv_3 = Conv2D(64, (1, 1), padding='same', strides=(1, 1))(relu_2)
        batch_norm_3 = BatchNormalization()(conv_3)
        relu_3 = Activation('relu')(batch_norm_3)

        conv_4 = Conv2D(64, (1, 1), padding='same', strides=(1, 1))(max_pool_1)
        batch_norm_4 = BatchNormalization()(conv_4)
        relu_4 = Activation('relu')(batch_norm_4)

        add = Add()([relu_3, relu_4])
        flatten = Flatten()(add)

        dense_1 = Dense(1024, activation='elu')(flatten)

        spot_prob = Dense((SQUARE_BOARD_SIZE + 1), kernel_regularizer=regularizers.l2(1e-4), activation='sigmoid', name="spot_prob")(dense_1)
        win_prob = Dense(1, kernel_regularizer=regularizers.l2(1e-6), activation='sigmoid', name="win_prob")(dense_1)

        model = Model(inputs=input_buffer, outputs=[spot_prob, win_prob])

        losses = {"spot_prob" : "categorical_crossentropy",   # Majorly use at muitl-classification -> Go problem's solution is find proper "one" location between multiple locations.
                  "win_prob"  : "binary_crossentropy"}        # 0, 1 Classification. For MSE, scale 0 ~ 1, not -1 ~ 1

        loss_weights = {"spot_prob" : 1e-3,
                        "win_prob"  : 1e-4}        

        return model , losses , loss_weights