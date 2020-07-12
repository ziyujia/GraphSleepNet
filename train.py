import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.io as scio
import shutil

import keras
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from model.GraphSleepNet import build_GraphSleepNet
from model.Utils import *
from model.DataGenerator import *

# # 1. Get configuration

# ## 1.1. Read .config file

# command line parameters -c -g
parser = argparse.ArgumentParser()
parser.add_argument("-c", type = str, help = "configuration file", required = True)
parser.add_argument("-g", type = str, help = "GPU number to use, set '-1' to use CPU", required = True)
args = parser.parse_args()
Path, cfgTrain, cfgModel = ReadConfig(args.c)

# set GPU number or use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = args.g
if args.g != "-1":
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    KTF.set_session(sess)
    print("Use GPU #"+args.g)
else:
    print("Use CPU only")

# ## 1.2. Analytic parameters

# [train] parameters
channels   = int(cfgTrain["channels"])
fold       = int(cfgTrain["fold"])
context    = int(cfgTrain["context"])
num_epochs = int(cfgTrain["epoch"])
batch_size = int(cfgTrain["batch_size"])
optimizer  = cfgTrain["optimizer"]
learn_rate = float(cfgTrain["learn_rate"])
lr_decay   = float(cfgTrain["lr_decay"])

# [model] parameters
dense_size            = np.array(str.split(cfgModel["Globaldense"],','),dtype=int)
conf_adj              = cfgModel["adj_matrix"]
GLalpha               = float(cfgModel["GLalpha"])
num_of_chev_filters   = int(cfgModel["cheb_filters"])
num_of_time_filters   = int(cfgModel["time_filters"])
time_conv_strides     = int(cfgModel["time_conv_strides"])
time_conv_kernel      = int(cfgModel["time_conv_kernel"])
num_block             = int(cfgModel["num_block"])
cheb_k                = int(cfgModel["cheb_k"])
l1                    = float(cfgModel["l1"])
l2                    = float(cfgModel["l2"])
dropout               = float(cfgModel["dropout"])

# ## 1.3. Parameter check and enable

# check optimizer（opt）
if optimizer=="adam":
    opt = keras.optimizers.Adam(lr=learn_rate,decay=lr_decay)
elif optimizer=="RMSprop":
    opt = keras.optimizers.RMSprop(lr=learn_rate,decay=lr_decay)
elif optimizer=="SGD":
    opt = keras.optimizers.SGD(lr=learn_rate,decay=lr_decay)
else:
    assert False,'Config: check optimizer'

# set l1、l2（regularizer）
if l1!=0 and l2!=0:
    regularizer = keras.regularizers.l1_l2(l1=l1, l2=l2)
elif l1!=0 and l2==0:
    regularizer = keras.regularizers.l1(l1)
elif l1==0 and l2!=0:
    regularizer = keras.regularizers.l2(l2)
else:
    regularizer = None
    
# Create save pathand copy .config to it
if not os.path.exists(Path['Save']):
    os.makedirs(Path['Save'])
shutil.copyfile(args.c, Path['Save']+"last.config")


# # 2. Read data and process data

# ## 2.1. Read data

ReadList = np.load(Path['data'],allow_pickle=True)

Fold_Num    = ReadList['Fold_Num']     # Samples of each fold [31]
# Out: list with same length
Fold_Data    = ReadList['Fold_Data']
Fold_Label   = ReadList['Fold_Label']
print("Read data successfully")

# ## 2.2. Read adjacency matrix

# get adj and calculate Chebyshev polynomials if Graph Learn is not used
if conf_adj!='GL':
    if   conf_adj=='1':
        adj=np.ones((channels,channels))
    elif conf_adj=='random':
        adj=np.random.rand(channels,channels)
    elif conf_adj=='topk' or conf_adj=='PLV' or conf_adj=='DD':
        adj=scio.loadmat(Path['cheb'])['adj']
    else: 
        assert False,'Config: check ADJ'
    L = scaled_Laplacian(adj)
    cheb_polynomials = cheb_polynomial(L, cheb_k)
else:
    cheb_polynomials = None

# ## 2.3. Add time context

Fold_Data    = AddContext(Fold_Data,context)
Fold_Label   = AddContext(Fold_Label,context,label=True)
Fold_Num_c  = Fold_Num + 1 - context
        
print('Context added successfully.')
print('Number of samples: ',np.sum(Fold_Num_c))

DataGenerator = kFoldGenerator(fold,Fold_Data,Fold_Label)

# # 3. Model training (cross validation)

# k-fold cross validation
all_scores = []
for i in range(fold):
    print('Fold #', i)
    
    # get i th-fold data
    train_data,train_targets,val_data,val_targets = DataGenerator.getFold(i)
    sample_shape = (context,train_data.shape[2],train_data.shape[3])   
    
    # build model
    model=build_GraphSleepNet(cheb_k, num_of_chev_filters, num_of_time_filters, time_conv_strides, cheb_polynomials, time_conv_kernel, 
                      sample_shape, num_block, dense_size, opt, conf_adj=='GL',GLalpha, regularizer, dropout)
    if i==0:
        model.summary()
        
    # train
    history = model.fit(
        x = train_data,
        y = train_targets,
        epochs = num_epochs,
        batch_size = batch_size,
        shuffle = True,
        validation_data = (val_data, val_targets),
        callbacks=[keras.callbacks.ModelCheckpoint(Path['Save']+'Best_model_'+str(i)+'.h5', 
                                                   monitor='val_acc', 
                                                   verbose=0, 
                                                   save_best_only=True, 
                                                   save_weights_only=False, 
                                                   mode='auto', 
                                                   period=1 )],
        verbose = 1)
    
    # save the final model
    model.save(Path['Save']+'Final_model_'+str(i)+'.h5')
    
    # Save training information
    if i==0:
        fit_acc=np.array(history.history['acc'])*Fold_Num[i]
        fit_loss=np.array(history.history['loss'])*Fold_Num[i]
        fit_val_loss=np.array(history.history['val_loss'])*Fold_Num[i]
        fit_val_acc=np.array(history.history['val_acc'])*Fold_Num[i]
    else:
        fit_acc=fit_acc+np.array(history.history['acc'])*Fold_Num[i]
        fit_loss=fit_loss+np.array(history.history['loss'])*Fold_Num[i]
        fit_val_loss=fit_val_loss+np.array(history.history['val_loss'])*Fold_Num[i]
        fit_val_acc=fit_val_acc+np.array(history.history['val_acc'])*Fold_Num[i]
        
    # Evaluate
    # Load weights of best performance
    model.load_weights(Path['Save']+'Best_model_'+str(i)+'.h5')
    val_mse, val_acc = model.evaluate(val_data, val_targets, verbose=0)
    print('Evaluate',val_acc)
    all_scores.append(val_acc)
    
    # Predict
    predicts = model.predict(val_data)
    AllPred_temp = np.argmax(predicts, axis=1)
    if i == 0:
        AllPred = AllPred_temp
    else:
        AllPred = np.concatenate((AllPred,AllPred_temp))
    
    # Fold finish
    print(128*'_')
    del model,train_data,train_targets,val_data,val_targets

# # 4. Final results

# Average training performance
fit_acc      = fit_acc/len(AllPred)
fit_loss     = fit_loss/len(AllPred)
fit_val_loss = fit_val_loss/len(AllPred)
fit_val_acc  = fit_val_acc/len(AllPred)

# print acc of each fold
print(128*'=')
print("All folds' acc: ",all_scores)
print("Average acc of each fold: ",np.mean(all_scores))

# Get all true labels
AllTrue = DataGenerator.getY_one_hot()
# Print score to console
print(128*'=')
PrintScore(AllTrue, AllPred)
# Print score to Result.txt file
PrintScore(AllTrue, AllPred, savePath=Path['Save'])

# Print confusion matrix and save
ConfusionMatrix(AllTrue, AllPred, classes=['W','N1','N2','N3','REM'], savePath=Path['Save'])

# Draw ACC / loss curve and save
VariationCurve(fit_acc, fit_val_acc, 'Acc', Path['Save'], figsize=(9, 6))
VariationCurve(fit_loss, fit_val_loss, 'Loss', Path['Save'], figsize=(9, 6))

