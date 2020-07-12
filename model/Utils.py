import configparser
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from scipy.sparse.linalg import eigs

# Read configuration file

def ReadConfig(configfile):
    config = configparser.ConfigParser()
    print('Config: ', configfile)
    config.read(configfile)
    cfgPath = config['path']
    cfgTrain = config['train']
    cfgModel = config['model']
    return cfgPath, cfgTrain, cfgModel

# Add context to the origin data and label

def AddContext(x, context, label=False, dtype=float):
    ret = []
    assert context%2==1, "context value error."
    
    cut = int(context/2)
    if label:
        for p in range(31):
            tData=x[p][cut:x[p].shape[0]-cut]
            ret.append(tData)
            #print(tData.shape)
    else:
        for p in range(31):
            tData=np.zeros([x[p].shape[0]- 2*cut,context,x[p].shape[1],x[p].shape[2]],dtype=dtype)
            for i in range(cut,x[p].shape[0]-cut):
                tData[i-cut]=x[p][i-cut:i+cut+1]
            #print(tData.shape)
            ret.append(tData)
    return ret

# Print score between Ytrue and Ypred
# savePath=None -> console, else to Result.txt

def PrintScore(true,pred,savePath=None,average='macro'):
    if savePath == None:
        saveFile = None
    else:
        saveFile=open(savePath+"Result.txt",'a+')
    # Main scores
    F1=metrics.f1_score(true,pred,average=None)
    print("Main scores:")
    print('Acc\tF1S\tKappa\tF1_W\tF1_N1\tF1_N2\tF1_N3\tF1_R',file=saveFile)
    print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f'% (metrics.accuracy_score(true,pred),
                                                             metrics.f1_score(true,pred,average=average),
                                                             metrics.cohen_kappa_score(true,pred),
                                                             F1[0],F1[1],F1[2],F1[3],F1[4]),
                                                             file=saveFile)
    # Classification report
    print("\nClassification report:",file=saveFile)
    print(metrics.classification_report(true,pred,target_names=['Wake','N1','N2','N3','REM']),file=saveFile)
    # Confusion matrix
    print('Confusion matrix:',file=saveFile)
    print(metrics.confusion_matrix(true,pred),file=saveFile)
    # Overall scores
    print('\n    Accuracy\t',metrics.accuracy_score(true,pred),file=saveFile)
    print(' Cohen Kappa\t',metrics.cohen_kappa_score(true,pred),file=saveFile)
    print('    F1-Score\t',metrics.f1_score(true,pred,average=average),'\tAverage =',average,file=saveFile)    
    print('   Precision\t',metrics.precision_score(true,pred,average=average),'\tAverage =',average,file=saveFile)
    print('      Recall\t',metrics.recall_score(true,pred,average=average),'\tAverage =',average,file=saveFile)
    # Results of each class
    print('\nResults of each class:',file=saveFile)
    print('    F1-Score\t',metrics.f1_score(true,pred,average=None),file=saveFile)
    print('   Precision\t',metrics.precision_score(true,pred,average=None),file=saveFile)
    print('      Recall\t',metrics.recall_score(true,pred,average=None),file=saveFile)
    if savePath != None:
        saveFile.close()
    return

# Print confusion matrix and save

def ConfusionMatrix(y_true, y_pred, classes, savePath, title=None, cmap=plt.cm.Blues):
    if not title:
            title = 'Confusion matrix'
    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    cm_n=cm
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Confusion matrix")
    print(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j]*100,'.2f')+'%\n'+format(cm_n[i, j],'d'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(savePath+title+".png")
    plt.show()
    return ax

# Draw ACC / loss curve and save

def VariationCurve(fit,val,yLabel,savePath,figsize=(9, 6)):
    plt.figure(figsize=figsize)
    plt.plot(range(1,len(fit)+1), fit,label='Train')
    plt.plot(range(1,len(val)+1), val, label='Val')
    plt.title('Model ' + yLabel)
    plt.xlabel('Epochs')
    plt.ylabel(yLabel)
    plt.legend()
    plt.savefig(savePath + 'Model_' + yLabel + '.png')
    plt.show()
    return

# compute \tilde{L}

def scaled_Laplacian(W):
    '''
    compute \tilde{L}
    ----------
    Parameters
    W: np.ndarray, shape is (N, N), N is the num of vertices
    ----------
    Returns
    scaled_Laplacian: np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis = 1))
    L = D - W
    lambda_max = eigs(L, k = 1, which = 'LR')[0].real
    return (2 * L) / lambda_max - np.identity(W.shape[0])

# compute a list of chebyshev polynomials from T_0 to T_{K-1}

def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}
    ----------
    Parameters
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)
    K: the maximum order of chebyshev polynomials
    ----------
    Returns
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}
    '''
    N = L_tilde.shape[0]
    cheb_polynomials = np.array([np.identity(N), L_tilde.copy()])
    for i in range(2, K):
        cheb_polynomials=np.append(cheb_polynomials,[2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2]],axis=0)
    return cheb_polynomials

