import sys
import os

#TensorFlow
import tensorflow as tf
from tensorflow.keras.layers import (Dense, Input, Lambda, Layer,
                                     Concatenate, Multiply,
                                     GlobalAveragePooling1D, GlobalMaxPooling1D, 
                                     TimeDistributed, Reshape, Flatten)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

#Other Models
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import network


# Makes the Input Layer from Input List
def make_input_layer(input_list, builder, **params):
    '''
    Input list is a string of the inputs of the dataloader
    '''
    input_layers = {}
    for i in input_list:
        shape = builder._info().features[i].shape
        if i == 'ecg':
            input_layers[i] = Input(shape=(shape[1],shape[0], ), dtype='float32', name=i)
            if 'single_lead' in params:
                if params['single_lead']:
                    input_layers[i] = Input(shape=(shape[1],1, ), dtype='float32', name=i)
        else:
            input_layers[i] = Input(shape=shape, dtype='float32', name=i)
                                
    return input_layers

# Makes ECG Input Shape
def make_cnn_params(builder, **params):
    ecg_shape = builder._info().features['ecg'].shape
    if params['is_by_lead']:
        params['input_shape'] = [ecg_shape[1], 1]
    else:
        params['input_shape'] = [ecg_shape[1], ecg_shape[0]]    
    if 'single_lead' in params:
        if params['single_lead']:
            params['input_shape'] = [ecg_shape[1], 1]
            
    return params

# Loads Stanford Model
def make_cnn(**params): 
    cnn = network.build_network(**params) 
    cnn = Model(cnn.inputs, cnn.layers[-4].output)
    
    return cnn

# Gets Pretrained Model if in params
def get_pretrained_model(**params):
    model = tf.keras.models.load_model(os.path.join(params['pretrain_path'], 'weights.h5'))
    model = model.get_layer('ecg_model')
    
    if type(params['tune_layers']) == int:
        for layer in model.layers[-params['tune_layers']:]:
            layer.trainable = True
    elif not params['tune_layers']:
        model.trainable = False
    elif params['tune_layers']:
        model.trainable = True
        
    for w in model.weights:
        w._handle_name = 'PT_' + w.name
        
    return model

# Makes the Network layer for ECGs
def make_ecg_model(ecg_input, builder, **params):

    #Update Parameters
    params = make_cnn_params(builder, **params)

    #Get CNN Model
    cnn = make_cnn(**params)

    #Run Through CNN
    if params['is_by_lead']:
        ecg = Lambda(lambda x: tf.expand_dims(tf.transpose(x, [0, 2, 1]), -1))(ecg_input)
        ecg_output = TimeDistributed(cnn)(ecg)

        if params['is_by_time']:
            ecg_output = Lambda(lambda x: tf.transpose(x, [0, 2, 1, 3]))(ecg_output) 
            ecg_output = Reshape(
                (ecg_output.shape[1], ecg_output.shape[2]*ecg_output.shape[3])
            )(ecg_output)
            ecg_output = TimeDistributed(Dense(int(params['ecg_out_size']/2)))(ecg_output)
            ecg_GAP = GlobalAveragePooling1D()(ecg_output)
            ecg_GMP = GlobalMaxPooling1D()(ecg_output)
            ecg_output = Concatenate()([ecg_GAP, ecg_GMP])         
        else:
            ecg_output = Lambda(lambda x: tf.transpose(x, [0, 2, 1, 3]))(ecg_output)  
            ecg_output = Reshape((ecg_output.shape[1], -1))(ecg_output)
            ecg_GAP = GlobalAveragePooling1D()(ecg_output)
            ecg_GMP = GlobalMaxPooling1D()(ecg_output)
            ecg_output = Concatenate()([ecg_GAP, ecg_GMP])
            ecg_output = Dense(params['ecg_out_size'])(ecg_output)


    else:
        ecg_output = cnn(ecg_input)
        print(params['ecg_out_size']/2)
        print(ecg_output)

        if params['is_by_time']:
            ecg_output = TimeDistributed(Dense(params['ecg_out_size']/2))(ecg_output)
            ecg_GAP = GlobalAveragePooling1D()(ecg_output)
            ecg_GMP = GlobalMaxPooling1D()(ecg_output)
            ecg_output = Concatenate()([ecg_GAP, ecg_GMP])
        else: 
            ecg_GAP = GlobalAveragePooling1D()(ecg_output)
            ecg_GMP = GlobalMaxPooling1D()(ecg_output)
            ecg_output = Concatenate()([ecg_GAP, ecg_GMP])
            ecg_output = Dense(params['ecg_out_size'])(ecg_output)
            
    #Make Model
    model = Model(ecg_input, ecg_output)
    model._name = 'ecg_model'
    
    return model

# Makes the Full Neural Network Model
def make_nn(input_list, builder, label_dict, **params):

    # Get Input Layers
    inputs = make_input_layer(input_list, builder, **params)
    
    # ECG Represetnation Learning
    if 'ecg' in input_list:
        if params['pretrain']:
            ecg_model = get_pretrained_model(**params)
        else:
            ecg_model = make_ecg_model(inputs['ecg'], builder, **params)
            
        ecg_output = ecg_model(inputs['ecg'])
        ecg_output = [ecg_output]
    else:
        ecg_output = []
                
                
    # Concatenate Features
    full_input =[v for k,v in inputs.items() if k != 'ecg']
    full_input = [tf.expand_dims(x, -1) if len(x.shape)==1 else x for x in full_input] + ecg_output
    if len(full_input) > 1:
        full_input = Concatenate()(full_input)
    else:
        full_input = full_input[0]

    #Run Through Neural Network
    if params['nn_layer_sizes'] is None:
        net = full_input
    else:
        for i, n in enumerate(params['nn_layer_sizes']):
            if i == 0: 
                net = Dense(n, activation='relu', 
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)
                           )(full_input)
                if params['is_multiply_layer']:
                    net2 = Dense(n, activation='relu', 
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)
                           )(full_input)
                    net = Multiply()([net, net2])
            else:
                net = Dense(n, activation='relu', 
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)
                           )(net)

    # Make Output Layer
    ## Reference: https://petamind.com/advanced-keras-custom-loss-functions/ for loss function
    outputs = []
    for i in label_dict.keys():
        #Label Shape
        output_size = builder._info().features[i].shape
        if len(output_size) == 0:
            # check if feature is a ClassLabel
            try:  
                output_size = builder._info().features[i].num_classes
                if type(label_dict[i]['reformat']) == int:
                    output_size = 2
                if label_dict[i]['reformat'] == 'extreme':
                    output_size = 2                 
                outputs.append(Dense(output_size, 
                                 activation = 'softmax',
                                 name = i)(net))
            except:
                if type(label_dict[i]['reformat']) == int:
                    output_size = 2
                    outputs.append(Dense(output_size, 
                                 activation = 'softmax',
                                 name = i)(net))
                elif 'bin' in label_dict[i]['reformat']:
                    output_size = int(label_dict[i]['reformat'].split(':')[1])
                    outputs.append(Dense(output_size, 
                                 activation = 'softmax',
                                 name = i)(net))
                    
                else:
                    output_size = 1
                    outputs.append(Dense(output_size, 
                                 activation = 'linear',
                                 name = i)(net))
        
        else:
            if 'bin' in label_dict[i]['reformat']:
                    output_size = int(label_dict[i]['reformat'].split(':')[1])
                    outputs.append(Dense(output_size, 
                                 activation = 'softmax',
                                 name = i)(net))
            else:
                outputs.append(Dense(output_size[0], name = i)(net))

    return Model(inputs, outputs) 

def make_model(input_list, builder, label_dict,**params):
    return make_nn(input_list, builder, label_dict, **params)
