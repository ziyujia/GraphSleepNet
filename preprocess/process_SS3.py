import scipy.io as sio
import random
import numpy as np
from DE_PSD import *

def ReadData(filename,pathF):
    '''
    Read DE or PSD from XXXX_psd_de.mat
    '''
    read_mat=sio.loadmat(pathF+filename+'_psd_de.mat')
    #read_data_psd=read_mat['psd']
    read_data_de=read_mat['de']
    #print(read_data_de.shape,end=' ')
    return read_data_de


def ReadLabel(filename,pathL):
    '''
    Read label from XXXX-Label.mat
    '''
    read_mat=sio.loadmat(pathL+filename+'-Label.mat')
    Label_lists=read_mat['label']
    #print(Label_lists.shape,end=' ')
    return Label_lists

def DE_PSD_a_File(filename,stft_para,save_dir='./data/DE_PSD/',data_dir='./data/SS3/'):
    '''
    compute PSD and DE of a file
    '''
    # Read origin data from XXXX-data.mat
    Data_mat=sio.loadmat(data_dir+filename+'-Data.mat')
    Data_lists=Data_mat['PSG']
    print(filename,Data_lists.shape,end='\t')
    data=Data_lists[0]
    print(data.shape,end='\n\t')
    
    # compute PSD\DE
    MYpsd = np.zeros([Data_lists.shape[0],26,len(stft_para['fStart'])],dtype=float)
    MYde  = np.zeros([Data_lists.shape[0],26,len(stft_para['fStart'])],dtype=float)
    for i in range(0,Data_lists.shape[0]):
        data=Data_lists[i]
        MYpsd[i],MYde[i]=DE_PSD(data,stft_para)

    print(MYpsd.shape,end=' ')

    # save to XXXX_psd_de.mat
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    sio.savemat(save_dir+filename+'_psd_de.mat', {'psd':MYpsd,'de':MYde})
    print('OK')
    return

def Prepare_K_Fold(Path, FileName, shuffle=False, seed=0, norm=True):
    '''
    make a set of data of 31-fold
    (to ensure independence of subjects, make every two files' data to one fold)
    --------
    Path:    include 'Path_Feature', 'Path_Label', 'Save_Name'
    shuffle: whether to disorder the file order(bool)
    seed:    random seed
    norm:    whether to normalize the data
    '''
    print('Path_Feature: ',Path['Path_Feature'])
    print('Path_Label:   ',Path['Path_Label'])
    
    # (optional) randomly scrambling data sets
    if shuffle:
        np.random.seed(seed)
        random.shuffle(FileName)

    Out_Data=[]
    Out_Label=[]
    
    Fold_Num=np.zeros([31],dtype=int)
    i = 0
    while i < 62:
        print('Fold #',int(i/2)+1,'\t',FileName[i],end=' ')
        
        FoldData = ReadData (FileName[i],Path['Path_Feature'])
        FoldLabel= ReadLabel(FileName[i],Path['Path_Label'])
        
        print(' ',FileName[i+1],end='  ')
        
        FoldData = np.row_stack((FoldData, ReadData (FileName[i+1],Path['Path_Feature'])))
        FoldLabel= np.row_stack((FoldLabel,ReadLabel(FileName[i+1],Path['Path_Label'])))
        
        Fold_Num[int(i/2)]=FoldLabel.shape[0]
        
        Out_Data.append(FoldData)
        Out_Label.append(FoldLabel)
        
        print(Out_Data[int(i/2)].shape,Out_Label[int(i/2)].shape)
        
        if i==0:
            All_Data  = FoldData
            All_Label = FoldLabel
        else:
            All_Data  = np.row_stack((All_Data, FoldData))
            All_Label = np.row_stack((All_Label, FoldLabel))

        i+=2
        
    # Data standardization
    if norm:
        mean = All_Data.mean(axis=0)
        std = All_Data.std(axis=0)
        All_Data -= mean
        All_Data /= std
        for i in range(31):
            Out_Data[i] -= mean
            Out_Data[i] /= std
     
    print('All_Data:  ', All_Data.shape)
    print('All_Label: ', All_Label.shape)
    return {
        'Fold_Num':   Fold_Num,
        'Fold_Data':  Out_Data,
        'Fold_Label': Out_Label
        }


if __name__ == "__main__":
    # define the path to load and save
    Path={
        'Path_Data'   : '../data/SS3/',    # XXXX-Data.mat  (Already exists)
        'Path_Label'  : '../data/SS3/',    # XXXX-label.mat (Already exists)
        'Path_Feature': '../data/DE_PSD/', # XXXX_psd_de.mat(Will generate)
        'Save_Name'   : '../data/SS3_DE_26_channels.npz'
    }
    # the parameters to extract DE and PSD
    stft_para={
        'stftn' :7680,
        'fStart':[0.5, 4,  8, 14, 31],
        'fEnd'  :[4,   8, 14, 31, 50],
        'fs'    :256,
        'window':30,
    }
    # the file No of the MASS SS3
    FileName=['01-03-0001', '01-03-0002', '01-03-0003', '01-03-0004', '01-03-0005', '01-03-0006', 
              '01-03-0007', '01-03-0008', '01-03-0009', '01-03-0010', '01-03-0011', '01-03-0012', 
              '01-03-0013', '01-03-0014', '01-03-0015', '01-03-0016', '01-03-0017', '01-03-0018', 
              '01-03-0019', '01-03-0020', '01-03-0021', '01-03-0022', '01-03-0023', '01-03-0024', 
              '01-03-0025', '01-03-0026', '01-03-0027', '01-03-0028', '01-03-0029', '01-03-0030', 
              '01-03-0031', '01-03-0032', '01-03-0033', '01-03-0034', '01-03-0035', '01-03-0036', 
              '01-03-0037', '01-03-0038', '01-03-0039', '01-03-0040', '01-03-0041', '01-03-0042', 
              '01-03-0044', '01-03-0045', '01-03-0046', '01-03-0047', '01-03-0048', '01-03-0050', 
              '01-03-0051', '01-03-0052', '01-03-0053', '01-03-0054', '01-03-0055', '01-03-0056', 
              '01-03-0057', '01-03-0058', '01-03-0059', '01-03-0060', '01-03-0061', '01-03-0062', 
              '01-03-0063', '01-03-0064']
    
    for file in FileName:
        print(file)
        DE_PSD_a_File(file, stft_para, Path['Path_Feature'], Path['Path_Data'])
    print("DE and PSD extraction complete.")

    # make fold packaged data
    ReadList = Prepare_K_Fold(Path, FileName)
    np.savez(
        Path['Save_Name'],
        Fold_Num   = ReadList['Fold_Num'],
        Fold_Data  = ReadList['Fold_Data'],
        Fold_Label = ReadList['Fold_Label']
        )
    print('Save OK')