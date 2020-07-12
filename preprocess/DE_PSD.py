import os
import numpy as np
import math
import scipy.io as sio
from scipy.fftpack import fft,ifft


def DE_PSD(data,stft_para):
    '''
    compute DE and PSD
    --------
    input:  data [n*m]          n electrodes, m time points
            stft_para.stftn     frequency domain sampling rate
            stft_para.fStart    start frequency of each frequency band
            stft_para.fEnd      end frequency of each frequency band
            stft_para.window    window length of each sample point(seconds)
            stft_para.fs        original frequency
    output: psd,DE [n*l*k]        n electrodes, l windows, k frequency bands
    '''
    #initialize the parameters
    STFTN=stft_para['stftn']
    fStart=stft_para['fStart']
    fEnd=stft_para['fEnd']
    fs=stft_para['fs']
    window=stft_para['window']

    WindowPoints=fs*window

    fStartNum=np.zeros([len(fStart)],dtype=int)
    fEndNum=np.zeros([len(fEnd)],dtype=int)
    for i in range(0,len(stft_para['fStart'])):
        fStartNum[i]=int(fStart[i]/fs*STFTN)
        fEndNum[i]=int(fEnd[i]/fs*STFTN)

    #print(fStartNum[0],fEndNum[0])
    n=data.shape[0]
    m=data.shape[1]

    #print(m,n,l)
    psd = np.zeros([n,len(fStart)])
    de = np.zeros([n,len(fStart)])
    #Hanning window
    Hlength=window*fs
    #Hwindow=hanning(Hlength)
    Hwindow= np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (Hlength+1)) for n in range(1,Hlength+1)])

    WindowPoints=fs*window
    dataNow=data[0:n]
    for j in range(0,n):
        temp=dataNow[j]
        Hdata=temp*Hwindow
        FFTdata=fft(Hdata,STFTN)
        magFFTdata=abs(FFTdata[0:int(STFTN/2)])
        for p in range(0,len(fStart)):
            E = 0
            #E_log = 0
            for p0 in range(fStartNum[p]-1,fEndNum[p]):
                E=E+magFFTdata[p0]*magFFTdata[p0]
            #    E_log = E_log + log2(magFFTdata(p0)*magFFTdata(p0)+1)
            E = E/(fEndNum[p]-fStartNum[p]+1)
            psd[j][p] = E
            de[j][p] = math.log(100*E,2)
            #de(j,i,p)=log2((1+E)^4)
    
    return psd,de


