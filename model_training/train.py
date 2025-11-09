import sys, os
import pickle

import numpy as np

#Tensorflow
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import tensorflow_probability as tfp

# Return Dictionary of Loss Functions
def make_loss_fn(label_dict):
    loss_fn = dict()
    for k, v in label_dict.items():
        loss_fn[k] = v['loss_fn']
            
    return loss_fn

# Return Dictionary of Metrics Functions
def make_metrics_fn(label_dict):
    metrics_fn = dict()
    for k, v in label_dict.items():
        if v['metrics'] in ['failure', 'censorship']:
            metrics_fn[k] = concordance_metric(v['metrics'], v['boundaries'])
        else:
            metrics_fn[k] = v['metrics']
            
    return metrics_fn

# Trains Neural Network Model
def train_nn(model, model_dir, train_data, val_data, label_dict, **params):
    #1. Save Dir
    output_weights_path = os.path.join(model_dir, 'weights.h5')
    output_weights_path
    
    #2. Compile Model
    ## Get Loss FN + weights
    loss_fn = make_loss_fn(label_dict)
    loss_weight = {k: v['loss_weight'] for k,v in label_dict.items()}
    
    ## Get Metrics
    metrics_fn = make_metrics_fn(label_dict)
    
    ## Compile
#     # Comment 3/14/2024, depending on the tf version, you'll need weighted_metrics instead of metrics to properly evaluate
    model.compile(
        loss=loss_fn,
        optimizer=optimizers.Adam(params['learning_rate']),
        metrics=metrics_fn,
        loss_weights = loss_weight
    )
    
#     model.compile(
#         loss=loss_fn,
#         optimizer=optimizers.Adam(params['learning_rate']),
#         weighted_metrics=metrics_fn,
#         loss_weights = loss_weight
#     )
    
    #3. Checkpoint + Callbacks
    checkpoint = ModelCheckpoint(
         output_weights_path,
         save_weights_only=False,
         save_best_only=True,
         verbose=1,
    )
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=int(params['epochs']/10), 
                                 verbose=1, mode='min', cooldown=1, min_lr=params['learning_rate']/100)
    earlyStop = EarlyStopping(monitor="val_loss", mode="min", patience=int(params['epochs']/5)) 
    callbacks = [checkpoint, reduceLR, earlyStop]
    
    #4. Fit Model
    model.fit(
        train_data,
        epochs=params['epochs'],
        validation_data=val_data,
        callbacks = callbacks
    )
    
    #5. Load best model
    model = tf.keras.models.load_model(output_weights_path, compile=False)
    
    return model

# Master Train Function
def train(model, model_dir, train_data, val_data, label_dict, **params):
    model = train_nn(model, model_dir, train_data, val_data, label_dict, **params)
    return model
