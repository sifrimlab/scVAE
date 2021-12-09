import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler



def step_lr(epoch, lr):
    """
    This function takes the current epoch and learning rate during training and
    divides the learning rate by 5 every 200 epochs. This function serves as
    the main argument for the LearningRateScheduler class. You don't need to
    initialize the function arguments
    """

    if (epoch != 0) & (epoch % 200 == 0): # Change every 200 epochs
        lr = lr / 5

    return lr
