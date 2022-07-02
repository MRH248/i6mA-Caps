from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D,Concatenate,concatenate, LSTM,LayerNormalization, BatchNormalization, Flatten, Dropout, Dense,SpatialDropout1D,SeparableConv1D
from tensorflow.keras.layers.experimental import RandomFourierFeatures
from tensorflow.keras.losses import poisson
 
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import keras
from keras.regularizers import l2
import numpy as np
from Bio import SeqIO
from keras import layers, models, optimizers
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask,CapsuleLayer_nogradient_stop

def listToString(s):  

    str1 = ""  
    
    # traverse in the string   
    for ele in s:  
        str1 += ele   
    
    # return string   
    return str1   

def encode(Strng):

    def encode_seq(s):
        Encode = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'U':[0,0,0,1],'T':[0,0,0,1],'N':[0,0,0,0]}
        return np.array([Encode[x] for x in s])


    elements1 = {}
    
    accumulator=0
    for row in Strng:

      my_hottie = encode_seq((row))
      out_final=my_hottie
      out_final = np.array(out_final)
      elements1[accumulator]=out_final
      accumulator += 1 
   
    X = list(elements1.items())
    an_array = np.array(X)
    an_array=an_array[:,1]
    transpose = an_array.T
    transpose_list = transpose.tolist()
    X=np.transpose(transpose_list)
    X=np.transpose(X)
    #y = np.array(data['label'], dtype = np.int32);
    
    return X






def CapsNet(weights):

    input_shape=(41,4)
    n_class=2
    routings=5
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='valid',kernel_initializer='random_uniform', activation='relu', name='conv1')(x)
    conv1 = Dropout(0.5)(conv1)
    conv2 = layers.Conv1D(filters=128, kernel_size=5, strides=1, padding='valid',kernel_initializer='random_uniform', activation='relu', name='conv2')(conv1)
    conv2 = Dropout(0.6)(conv2)
    conv2 = layers.Conv1D(filters=128, kernel_size=7, strides=1, padding='valid',kernel_initializer='random_uniform', activation='relu')(conv2)
    conv2 = Dropout(0.4)(conv2)
    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv2, dim_capsule=8, n_channels=50, kernel_size=10, strides=2, padding='valid', dropout=0.2)
    
    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=10, num_routing=routings,name='digitcaps', kernel_initializer='random_uniform')(primarycaps)    
    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps) 
  

    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(256, activation='relu', input_dim=10*n_class))
    decoder.add(layers.Dropout(0.4))
    dec_1 = decoder.add(layers.Dense(128, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape))

    # Models for training and evaluation (prediction)
    model = models.Model([x],[out_caps, decoder(masked)]) #masked_by_y

    # manipulate modelTACTTCTGGTACGAGTGATTATCTTTTTACCGGGAGTCGGGG

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked)]) #masked_by_y
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 10))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))

    model_weights=weights
    train_model.load_weights(model_weights)

    # manipulate model

    return train_model, eval_model, manipulate_model


def prediction(model1,X):
    Predict=[]
    predictions,score = model1.predict([X])
    decision = []
    for i in range(len(predictions)):
        if predictions[i,0] > predictions[i,1]:
            decision.append(predictions[i,0])
        else:
            decision.append(predictions[i,1])
    pred_y = decision 
    n=len(X)
    tempLabel = [0]*n
    for i in range(len(decision)):
        if pred_y[i] < 0.5:
            tempLabel[i] = 0;
        else:
            tempLabel[i] = 1;

  
    return tempLabel




Testsequences = [] 
for record in SeqIO.parse("/home/mobeen/Capsule_net/Server/fasta.fa", "fasta"):
    Testsequences.append(record.seq.upper())
    n=len(record)

weights='Athaliana.h5'

X=encode(Testsequences)
model,eval_model,manipulate_model=CapsNet(weights)
Prediction=prediction(eval_model,X)

