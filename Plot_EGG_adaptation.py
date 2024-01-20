# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 15:42:42 2021

@author: seany
"""
import pandas as pd
import numpy as np
import scipy as sp
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.interpolate import interp1d
from matplotlib.offsetbox import AnchoredOffsetbox
import re
from scipy.interpolate import CubicSpline
def read_egg_v2(file,channels=1,header=7,rate=32):
    """
    Data import for EGGv2. 
    
    Parameters
    ----------
    file : string
        filepath to data from EGG recording.
    channels : int, optional
        Number of channels, used for parsing data. The default is 1.
    header : int, optional
        Number of leading lines to skip simple transmission msgs. The default is 7.
    rate : float, optional
        sampling rate off measurement. The default is 32.

    Returns
    -------
    datlist : list of 2xN numpy arrays 
        Each array indicates one channel of the recording, where [0,:] are timestamps and [1,:] are measurements
    """
    dat=pd.read_csv(file,header=header)
    datarray=[]
    #samples per second
    #rate=32
    #rate=8
    dup_i=0
    for row in range(len(dat)):
        duparray=np.array([False])
        #Remove duplicates in array from ack failure during data transmission
        if row > 0:
            duparray=np.array(dat.iloc[row,1:-1])==np.array(dat.iloc[row-1,1:-1]) 
        if all(duparray): 
            dup_i+=1
        else:
    #        print(row)
            for column in range(len(dat.iloc[row])):            
                if column == 0:
                    element=dat.iloc[row,column]
                    mod=element.split('>')
                    datarray.append(int(mod[1]))
                if column > 0 and column < 30:
                    datarray.append(int(dat.iloc[row,column]))
    datarray=np.array(datarray)
    convarray=[]
    timearray=np.array(np.arange(len(datarray))/rate)
    for ele in datarray:
        if ele<2**15:
            convarray.append(0.256*ele/(2**15))
        else:
            convarray.append(0.256*(((ele-2**16)/(2**15))))
    voltarray=np.array(convarray)
    voltarray.shape
    timearray.shape
    size=np.int(voltarray.size/channels)
    print(size)
    reshaped_volt=np.reshape(voltarray,(size,channels))
    reshaped_time=np.reshape(timearray,(size,channels))
    
    datlist=[]
    for num in range(channels):
        datlist.append(np.array([reshaped_time[:,num],reshaped_volt[:,num]*1000])) #have to convert to mV
    return datlist

def egg_interpolate(dat,rate=62.5,start_value=0,end_value=0):
    f=interp1d(dat[0,:],dat[1,:])
    if start_value==0: start_value=dat[0,:].min()
    if end_value==0:end_value=dat[0,:].max()
    tfixed=np.arange(start_value,end_value, 1/rate)
    return tfixed, f(tfixed)

def egg_filter(dat,rate=32,freq=[0,0.1],order=3,ncomb=0,debug=0):
    """
    Function which filters data using a butterworth filter
    Parameters
    ----------
    dat : List of 2 np arrays
        List of 2 np arrays where first array are timestamps and 2nd array is values
    rate : sampling rate in seconds, optional
        Sampling rate in seconds, used for interpolation of data prior to filtering. The default is 32.
    freq : List, optional
        Bandpass filter frequency. The default is [0,0.1].
    order : int, optional
        Order of butterworth filter generated for filtering. The default is 3.
    ncomb : float
        frequency in hrz of notch comb filter
    Returns
    -------
    fdata: numpy array of 2xN.
        1st index is columns, 2nd is rows. 1st column are timestamps and 2nd column is filtered data.

    """
    fn=rate/2
    wn=np.array(freq)/fn
#    wn[0]=np.max([0,wn[0]])
    wn[1]=np.min([.99,wn[1]])
#    print(wn)
    f=interp1d(dat[0,:],dat[1,:])
#    print(f)
    start_value=dat[0,:].min()
    end_value=dat[0,:].max()
    tfixed=np.arange(start_value,end_value, 1/rate)
    sos=sig.butter(order,wn,btype='bandpass',output='sos')
    filtered=sig.sosfiltfilt(sos,f(tfixed))
#    b,a=sig.butter(order,wn,btype='bandpass')
    
#    if debug == 1:
#        w,h=sig.freqs(b,a)
#        fig,ax=plt.subplots(figsize=(5,5))
#        ax.semilogx(w,20*np.log10(abs(h)))
#        ax.vlines(wn,ymin=-10,ymax=10)
    
#    filtered=sig.filtfilt(b,a,f(tfixed),method='pad')
    if ncomb!=0:
        if not isinstance(ncomb, list):
            ncomb=[ncomb]
        for ele in ncomb:
            c,d=sig.iircomb(ele/rate, 3)
            filtered=sig.filtfilt(c,d,filtered)
    
    fdata=np.array([tfixed,filtered])
    return fdata

def egg_fft(dat,rate=32,xlim=[-5,5],ylim=0):
    f=interp1d(dat[0,:],dat[1,:])
    start_value=dat[0,:].min()
    end_value=dat[0,:].max()
    tfixed=np.arange(start_value,end_value, 1/rate)
    fftdat=fftpack.fft(f(tfixed))
    freqs=fftpack.fftfreq(len(f(tfixed)))*rate*60
    fig, ax = plt.subplots()
    ax.stem(freqs, np.abs(fftdat))
    ax.set_xlabel('Frequency in 1/mins')
    ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
    ax.set_xlim(xlim)
    if ylim!=0: ax.set_ylim(ylim)
    return freqs,fftdat

def egg_powerspectrum(dat, rate=62.5,vlines=[]):
    x,y=egg_interpolate(dat,rate=rate)
    f, pden=sig.periodogram(y,fs=rate)
    figP=plt.figure()
    ax_P=figP.add_subplot(111)
    ax_P.loglog(f, pden)
    ax_P.set_ylim([1e-7,1e6])
    ax_P.set_xlim([.001,20])
    ax_P.set_ylabel('Power')
    ax_P.set_xlabel('Frequency (Hz)')
    ax_P.vlines(vlines,ymin=0,ymax=1e10,linewidth=1,color='black')
    return figP, ax_P

def read_egg_v3(file,header=0,rate=62.5,scale=150,error=0):
    """
    This is a function which uses pandas to read in data recorded from EGG V3 and transmitted to a board using
    RFStudio7. 
    
    file : filepath of the target txt file
    header : Number of lines to skip
    rate : Sampling rate in samples/second per channel set on the ADS131m8
    scale : +- scale in mV 
    error : returns data with CRC errors. Default is 0 so those are stripped
    
    output: Pandas data frame with the following information:
        .realtime : realtime from RFStudio when packet was received
        .misc : RF Studio output, not useful
        .packet : packet number, set from EGGv3, ranges from 0 to 65535 (unit16). Roll over if higher
        .msg : str of packet recieved
        .rssi : RSSI of packet, also includes CRC error
        'Channel n': Channels of recording data in mV, n is from 0 to 7
        .counter : absolute renumbered packets (without overflow)
        .timestamps : timesamples calculated from sampling rate and absolute timer
        .SPI : SPI Status (first packet of msg)
    
    """
    dat=pd.read_csv(file, header=header, dtype = str, delimiter='|', names=['realtime','misc','packet','msg','rssi'])
    dat=dat[~dat.rssi.str.contains('error')]
    dat=dat[dat.misc.str.contains('16')]
    dat=dat.reset_index(drop=True)
    dat_col=dat.msg
    hexdat=dat_col.str.split(' ') #Return list of splits based on spaces in msg
    serieslist=[]
    for k,ele in enumerate(hexdat):
        if len(ele) == 23: #Only select those that have the correct length
            vlist=[]
            for i in range(0,10):
                n=i*2+2
                value= ''.join(['0x',ele[n],ele[n-1]])
                hvalue=int(value,16)
                if i==0:
                    vlist.append(hvalue) #append hex code
                else:    
                    if hvalue<2**15:
                        vlist.append(scale*float(hvalue)/(2**15))
                    else:
                        vlist.append(scale*(((float(hvalue)-2**16)/(2**15))))
        else:
#            print('Line Error!'+str(k))
#            print(ele)
            vlist=[] #add empty list on error
        serieslist.append(vlist)
    collist=['SPI']
    for i in range(8): collist.append('Channel '+str(i)) #make channel list name
    collist.append('CRC')
    datalist=pd.DataFrame(serieslist,columns=collist)
    print(datalist)
    print(dat)
    fulldat=pd.concat((dat,datalist),axis=1)
    print(fulldat)
    counter=fulldat.packet.astype(int)
    new_counter=[0]
    for j,ele in enumerate(counter[1:]): #Renumbered counter - note this will give an error if you accidentally miss the 0/65535 packets
        step=counter[j+1]-counter[j]
#       if step != -65535:
        if step > 0:
            new_counter.append(step+new_counter[j])
#       elif step < 0:
#            new_counter.append(new_counter[j])
        else:
            new_counter.append(65536-counter[j]+counter[j+1]+new_counter[j])
            print('flip', step, 65536-counter[j]+counter[j+1])
#            new_counter.append(1+new_counter[j])
    tarray=np.array(new_counter)*1/rate
    abscounterseries=pd.Series(new_counter,name='counter')
    tseries=pd.Series(tarray,name='timestamps')
    
    fulldat=pd.concat((fulldat,abscounterseries,tseries),axis=1)
    noerror=~fulldat.rssi.str.contains('error') # Gives rows without crc error
    if error: 
        return fulldat # return non-crc error 
    else:
        return fulldat[noerror]
#    hexdat.dropna() #drop out of range NaNs without shifting indicies

def read_egg_v3_burst(file,header=0,rate=62.5,scale=150,error=0, burst_size=5,start_sleep=240,sleep_ping=1,sleep_timer=2):
    """
    This is a function which uses pandas to read in data recorded from EGG V3 and transmitted to a board using
    RFStudio7. 
    
    file : filepath of the target txt file
    header : Number of lines to skip
    rate : Sampling rate in samples/second per channel set on the ADS131m8
    scale : +- scale in mV 
    error : returns data with CRC errors. Default is 0 so those are stripped
    
    output: Pandas data frame with the following information:
        .realtime : realtime from RFStudio when packet was received
        .misc : RF Studio output, not useful
        .packet : packet number, set from EGGv3, ranges from 0 to 65535 (unit16). Roll over if higher
        .msg : str of packet recieved
        .rssi : RSSI of packet, also includes CRC error
        'Channel n': Channels of recording data in mV, n is from 0 to 7
        .counter : absolute renumbered packets (without overflow)
        .timestamps : timesamples calculated from sampling rate and absolute timer
        .SPI : SPI Status (first packet of msg)
    
    """
    dat=pd.read_csv(file, header=header, dtype = str, delimiter='|', names=['realtime','misc','packet','msg','rssi'])
    dat=dat[~dat.rssi.str.contains('error')]
    dat=dat[dat.misc.str.contains('16')]
    dat=dat.reset_index(drop=True)
    dat_col=dat.msg
    hexdat=dat_col.str.split(' ') #Return list of splits based on spaces in msg
    serieslist=[]
    for k,ele in enumerate(hexdat):
        if len(ele) == 23: #Only select those that have the correct length
            vlist=[]
            for i in range(0,10):
                n=i*2+2
                value= ''.join(['0x',ele[n],ele[n-1]])
                hvalue=int(value,16)
                if i==0:
                    vlist.append(hvalue) #append hex code
                else:    
                    if hvalue<2**15:
                        vlist.append(scale*float(hvalue)/(2**15))
                    else:
                        vlist.append(scale*(((float(hvalue)-2**16)/(2**15))))
        else:
#            print('Line Error!'+str(k))
#            print(ele)
            vlist=[] #add empty list on error
        serieslist.append(vlist)
    collist=['SPI']
    for i in range(8): collist.append('Channel '+str(i)) #make channel list name
    collist.append('CRC')
    datalist=pd.DataFrame(serieslist,columns=collist)
    print(datalist)
    print(dat)
    fulldat=pd.concat((dat,datalist),axis=1)
    print(fulldat)
    counter=fulldat.packet.astype(int)
    new_counter=[0]
    for j,ele in enumerate(counter[1:]): #Renumbered counter - note this will give an error if you accidentally miss the 0/65535 packets
        step=counter[j+1]-counter[j]
#       if step != -65535:
        if step > 0:
            new_counter.append(step+new_counter[j])
#       elif step < 0:
#            new_counter.append(new_counter[j])
        else:
            new_counter.append(65536-counter[j]+counter[j+1]+new_counter[j])
            print('flip', step, 65536-counter[j]+counter[j+1])
#            new_counter.append(1+new_counter[j])
    
    tarray=[]
    for number in new_counter:
        burst_time=np.floor((number)/(burst_size+sleep_ping))*sleep_timer
        #burst_time=(number)
        #burst_time=0
        packet_time=((number-start_sleep) % (burst_size+sleep_ping))*1/rate
        #packet_time=0
        tarray.append(float(burst_time)+packet_time)
    #tarray=np.array(new_counter)*1/rate
    
    abscounterseries=pd.Series(new_counter,name='counter')
    tseries=pd.Series(tarray,name='timestamps')   
    fulldat=pd.concat((fulldat,abscounterseries,tseries),axis=1)
    noerror=~fulldat.rssi.str.contains('error') # Gives rows without crc error
    if error: 
        return fulldat # return non-crc error 
    else:
        return fulldat[noerror]
#    hexdat.dropna() #drop out of range NaNs without shifting indicies

def migut_burst_interpolate(migut_data,rate=62.5):
    new_data=pd.DataFrame(np.arange(0,max(migut_data['timestamps']),1/rate),columns=['timestamps'])
    
    #migut_data['Synctime']=migut_data['Datetime'].apply(lambda x: ((x-t0).total_seconds()))
     
    migut_channels=[x for x in migut_data.columns if re.match('Channel*',x)] #Us regex to find Channel * column names
    for chan in migut_channels:
        f=CubicSpline(migut_data['timestamps'],migut_data[chan])
        interpolated_data=f(new_data['timestamps'])
        new_data[chan]=interpolated_data
    return new_data

#%%    
def signalplot(dat,xlim=(0,0,0),spacer=0,vline=[],line_params= ['black', 5, 'dashed'],freq=1,order=3,rate=62.5, title='',skip_chan=[],figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={}):
    """
    Function to plot all channels in dataframe following data import using read_egg_v3

    Inputs:
        sig: Dataframe containing time data in "timestamps" column and "Channel n" where n is channel number
        xlim: list of 2 elements taking time range of interest. Default behavior is to full timescale
        spacer: Spacing between plots, and their scaling. Default behavior is spaced on max y value in a channel
        freq: frequency list in Hz, 2 element list, for bandpass filtering. Default is no filtering
        order: order of butter filter used in filtering
        rate: sampling rate of data, for filtering
        title: title label of plot, default is none 
        vline: list of float marking lines for xvalues, usually for fast visulation/measurement
        skip_chan: list of channels to skip, default none. 
        figsize: tuple of figure size dimensions passed to matplotlib.figure, default 10,20
        ncomb: comb frequency hz, passed to egg filter
        
    Outputs:
        fig_an: figure instance of plot (from matplotlib)
        ax_an: axis instance of plot
        Outarray.T: exported filtered data
    Changelog:
        2023.09.29 - Added color-dict functionality
        2023.08.18 - Added rescaling functionality
    """
    x=dat[time].to_numpy()
    outarray=[]
    if len(skip_chan)>0:
        skip_chan_type=list(map(type,skip_chan))
#        skip_chan_len=list(map(len,skip_chan))
        if int in skip_chan_type or (len(skip_chan[0])<5): #This line preserves easy behavior of calling 1,2,3,a,b etc for skip_chan when not using column names
            skip_chan=list(map(lambda x: "Channel "+str(x),skip_chan))
    if len(labels) == 0: labels=dat.columns
    if freq==1: outarray.append(x)
    plt.rcParams['font.size']=textsize
    fig_an, ax_an = plt.subplots(figsize=figsize) 
    # we make only 1 axis instance and we will manually displace the plots below
    if len(xlim)==2:
        ax_an.set_xlim(xlim[0],np.min([xlim[1],x.max()]))
    else:
        ax_an.set_xlim([x.min(),x.max()])
        xlim=[x.min(),x.max()]
    xloc=ax_an.get_xlim()[0]
    ax_an.spines['right'].set_visible(False)
    ax_an.spines['top'].set_visible(False)
    ax_an.spines['left'].set_visible(False)
    ax_an.xaxis.set_ticks_position('none')
    ax_an.xaxis.set_ticks_position('bottom')
    ax_an.set_yticks([])
    ax_an.set_xlabel('Time (s)')
    xsize=ax_an.get_xlim()[1]-ax_an.get_xlim()[0]   

    loc=np.logical_and(x>xlim[0],x<xlim[1])
    space=0
    if spacer == 0: #this is to automatically set the spacing we want between the 
        distarr=[]
        for i,column in enumerate(labels):
#            if column.startswith('Channel') and not(int(column[-2:]) in skip_chan):
            if column.startswith('Channel') and not(column in skip_chan):
                y=dat[column].to_numpy()                
                if freq == 1:
                    if Normalize_channels: y=y/(y[loc].max()-y[loc].min())
                    distance=y[loc].max()-y[loc].min()
                else:
                    mod=egg_filter(np.array([x,y]),freq=freq,rate=rate,order=order,ncomb=ncomb)

                    loc2=np.logical_and(mod[0,:]>xlim[0],mod[0,:]<xlim[1])
                    if Normalize_channels: mod[1,:]=mod[1,:]/(mod[1,loc2].max()-mod[1,loc2].min()) # Need to divide by whole range
                    distance=mod[1,loc2].max()-mod[1,loc2].min()
                
                distarr.append(distance)
        distarr=np.array(distarr)
#        print(distarr)
        spacer=distarr.max()*1.1    
    column_list=['Synctime']
    for i,column in enumerate(labels):
#        if column.startswith('Channel') and not(int(column[-2:]) in skip_chan):
        if column.startswith('Channel') and not(column in skip_chan):
            y=dat[column].to_numpy()
            column_list.append(column)

            if freq == 1:
                if Normalize_channels: y=y/(y[loc].max()-y[loc].min())
                if column in color_dict:
                    ax_an.plot(x, y-y[loc].mean()+space,color_dict[column])
                else:
                    ax_an.plot(x, y-y[loc].mean()+space)
                if points:
                   ax_an.plot(x, y-y[loc].mean()+space,'ro') 
                print('plotted!')
                outarray.append(y)
            else:
                mod=egg_filter(np.array([x,y]),freq=freq,rate=rate,order=order,ncomb=ncomb)
                loc2=np.logical_and(mod[0,:]>xlim[0],mod[0,:]<xlim[1])
                if Normalize_channels: 
                    mod[1,:]=mod[1,:]/(mod[1,loc2].max()-mod[1,loc2].min())
                if len(outarray)==0: outarray.append(mod[0,:].squeeze())
                if column in color_dict:
                    ax_an.plot(mod[0,loc2], mod[1,loc2]+space,color_dict[column])
                else:
                    ax_an.plot(mod[0,loc2], mod[1,loc2]+space)
                if points:
                    ax_an.plot(mod[0,loc2], mod[1,loc2]+space,'ro')
                outarray.append(mod[1,:].squeeze())
#            print(dat[column].name)
            if not hide_y:
                if dat[column].name in name_dict:
                    ax_an.text(ax_an.get_xlim()[0]-xsize/40,space,name_dict[dat[column].name],ha='right')
                else:
                    ax_an.text(ax_an.get_xlim()[0]-xsize/40,space,dat[column].name,ha='right')
            space+=spacer
#            print(space)
    if len(vline) != 0:
        ax_an.vlines(vline,ymin=0-spacer/2, ymax=space-spacer/2,color=line_params[0],linewidth=line_params[1],linestyle=line_params[2])
    if len(hline) != 0:
        ax_an.hlines(hline,xmin=xlim[0],xmax=xlim[1],linewidth=5,color='black',linestyle='dashed')
    ax_an.set_ylim(0-spacer,space)
    ax_an.set_title(title)

    if not Normalize_channels: #Only show voltage if everything is not rescaled
        ax_an.vlines(xlim[0],ymin=0-3*spacer/4,ymax=0-spacer/2,linewidth=10,color='black')
        ax_an.text(xlim[0]+xsize/40,0-5/8*spacer,str(np.round(spacer*1/4,decimals=2))+' mV',ha='left') 
        
#    add_scalebar(ax_an,hidex=False,matchy=True)
    outarray=np.array(outarray)
    loc_out=np.logical_and(outarray[0,:]>xlim[0],outarray[0,:]< xlim[1])
    outarray=outarray[:,loc_out]
    
    
    if output=="PD": 
        outarray=pd.DataFrame(outarray.T,columns=column_list)
    else: 
        outarray=outarray.T
    return fig_an,ax_an,outarray
#%%
def heatplot(dat, xlim=(0, 0, 0), spacer=0, vline=[], freq=1, order=3, rate=62.5, title='', skip_chan=[], figsize=(10, 10), textsize=16, vrange=[0, 0, 0], interpolation='bilinear', norm=True):
    plt.rcParams['font.size'] = textsize
    fig_an, ax_an = plt.subplots(figsize=figsize)
    x = dat.timestamps.to_numpy()
    arraylist = []
    
    # Added to this function, initialize active channels
    active_channels = []
    
    if len(xlim) == 2:
        ax_an.set_xlim(xlim[0], np.min([xlim[1], x.max()]))
    else:
        ax_an.set_xlim([x.min(), x.max()])
        xlim = [x.min(), x.max()]
        
    for i, column in enumerate(dat.columns):
        if column.startswith('Channel') and not(int(column[-2:]) in skip_chan):
            y = dat[column].to_numpy()
            if freq == 1:
                xf, yf = egg_interpolate(np.array([x, y]), rate=rate)
            else:
                d = np.array([x, y])
                mod = egg_filter(d, freq=freq, rate=rate, order=order)
                xf = mod[0, :]
                yf = mod[1, :]
            arraylist.append(yf)
            
            # Add active channel number to list
            active_channels.append(int(column[-2:]))
            
    datlist = np.array(arraylist)
    if len(xlim) == 2:
        loc2 = np.logical_and(xf > xlim[0], xf < xlim[1])
        datlist = datlist[:, loc2]
    if norm:
        datlist = np.absolute(datlist)
        
    if len(vrange) == 2:
        colors = ax_an.imshow(np.flip(datlist, axis=0), aspect='auto', extent=[xlim[0], xlim[1], -0.5, len(active_channels)-0.5], cmap='jet', vmin=vrange[0], vmax=vrange[1], interpolation=interpolation)
    else:
        colors = ax_an.imshow(np.flip(datlist, axis=0), aspect='auto', extent=[xlim[0], xlim[1], -0.5, len(active_channels)-0.5], cmap='jet', interpolation=interpolation)  
    
    # Set the y-ticks and y-tick labels
    ax_an.set_yticks(range(len(active_channels)))
    ax_an.set_yticklabels(active_channels)
    
    ax_an.set_xlabel('Time (s)')
    ax_an.set_ylabel('Channel Number')
    
    cbar = fig_an.colorbar(colors, ax=ax_an)
    cbar.set_label('Electrical Activity (mV)', labelpad=10)
    return fig_an, ax_an, datlist



def rssiplot(dat,xlim=[0,0,0],figsize=(5,5),ylim=[-100,-20],textsize=16):
    plt.rcParams['font.size']=textsize
    x=dat.timestamps.to_numpy()
    y=np.asarray(dat.rssi.to_numpy(),dtype=np.float64)
    fig_an, ax_an = plt.subplots(figsize=figsize)
    if len(xlim)==2:
        ax_an.set_xlim(xlim[0],np.min([xlim[1],x.max()]))
    else:
        ax_an.set_xlim([x.min(),x.max()])
        xlim=[x.min(),x.max()]            
    ax_an.set_ylim(ylim)
#    ax_an.set_yticks([-100,-80,-60,-40,-20])
    ax_an.plot(x,y,'ro',markersize=.5)
    ax_an.set_ylabel("RSSI (dB)")
    ax_an.set_xlabel("Time (s)")
    return fig_an,ax_an

#%%
def egg_signalfreq(dat,rate=62.5,freqlim=[1,10],ylim=0,mode='power',ylog=False,xlog=False,clip=False,mmc=False,labels=[],figsize=(10,20),title='',vline=[],vline_color='black',textsize=12,name_dict={}):
    '''

    Parameters
    ----------
    dat : numpy arrray 
        packaged as a array of Nx9 size, where first column is time and then remaining columns are channel data
        
    rate : float
        The sampling rate in samples/seconds. The default is 62.5.
    xlim : list of 2 floats 
        The frequency range in samples/minute  The default is [1,10].
    ylim : list of 2 floats, optional 
        Set the yrnage of the exported graph
    mode : str, optional
       Mode of operation:
           'power' : power spectrum
           'fft' : fft

    Returns
    -------
    fig : Figure instance of plot
    ax : Axis instance of plot
    array : np array containing 9XN where first row is frequncies and remainder are power/intensities

    '''
    plt.rcParams.update({'font.size':textsize})
#    functionlist=[]
    dlist=[]    
#    for i in range(1:7): functionlist.
#    f=interp1d(dat[0,:],dat[1,:])
#    start_value=dat[:,0].min()
#    end_value=dat[:,0].max()
#    tfixed=np.arange(start_value,end_value, 1/rate)
    namelist=[]
    if isinstance(dat,pd.DataFrame):
        namelist=dat.columns[1:]
        dat=dat.to_numpy()

    if mode == 'fft':
        for i in range(1,dat.shape[1]):
            fftdat=fftpack.fft(dat[:,i])
            dlist.append(fftdat)
        x=fftpack.fftfreq(len(dat[:,0]))*rate*60 #Conversion to samples/minute for ease of use with slow waves
        
    if mode == 'power':
        for i in range(1,dat.shape[1]):
            x, pdat=sig.periodogram(dat[:,i],fs=rate, nfft=len(dat)*2)
            dlist.append(pdat)
        if mmc:
            x=x*3600
        else:    
            x=x*60 #Conversion to samples/minute for ease of use with slow waves

    loc = (x > freqlim[0]) & (x < freqlim[1]) #Note we are setting location to only plot data without the xlim range
    # Adjust for single channel case
    fig, ax = plt.subplots(nrows=max(1, dat.shape[1]-1), figsize=figsize)
    single_channel = dat.shape[1] == 2

    for i in range(1, dat.shape[1]):
        if single_channel:
            current_ax = ax
        else:
            current_ax = ax[dat.shape[1]-1-i]
        current_ax.ticklabel_format(axis='y', scilimits=(0, 0))
        current_ax.stem(x[loc], np.abs(dlist[i-1])[loc])
        # Handle labels
        label = 'Channel ' + str(i-1)
        if len(namelist) == dat.shape[1]-1:
            label = name_dict.get(namelist[i-1], namelist[i-1])
        elif len(labels) == dat.shape[1]-1:
            label = name_dict.get(labels[i-1], labels[i-1])
        current_ax.set_ylabel(label, size=16)
        current_ax.ticklabel_format(axis='y', style='sci')
        # Additional plot settings
        if len(vline) != 0:
            ymax = current_ax.get_ylim()[1] * 1.5
            ymin = current_ax.get_ylim()[0]
            if ylog: ymax = ymax / 1.5 * 10
            current_ax.vlines(vline, ymin, ymax, linewidth=2, color=vline_color, linestyle='dashed')
        if i != 1: 
            current_ax.set_xticklabels([])  # Hide all labels other than bottom label
        if ylog: 
            current_ax.set_yscale('log')
        if xlog: 
            current_ax.set_xscale('log')
    if not single_channel:
        ax[dat.shape[1]-9].set_title(title)
        if mmc:
            ax[dat.shape[1]-2].set_xlabel('Frequency (1/hr)')
        else:     
            ax[dat.shape[1]-2].set_xlabel('Frequency (1/mins)', size=16)  # Bottom graph label
    else:
        if mmc:
            ax.set_xlabel('Frequency (cycles/hr)')
        else:    
            ax.set_xlabel('Frequency (cycles/min)', size=16)
    fig.align_ylabels()
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    freq_power_chan = np.concatenate(([x], dlist), axis=0)
    if clip: 
        freq_power_chan = freq_power_chan[:, loc]
    return fig, ax, freq_power_chan

def egg_freq_heatplot(dat, rate=62.5, xlim=[0,10000],seg_length=500,freq=[0.02,0.2],freqlim=[1,10],vrange=[0],figsize=(10,20),interpolation='bilinear',n=10, intermediate=False,max_scale=.4,norm=True,time='timestamps'):
    '''

    Parameters
    ----------
    dat : Pandas Dataframe
        Dataframe producted from read_egg_v3.
    rate : float
        The sampling rate in samples/seconds. The default is 62.5.
    xlim : list of 2 floats, optional
        Range in seconds that will be plotted in time. The default is [0,10000].
    seg_length : float, optional
        Time in seconds that will be used to calculate frequency. The default is 500.
    freq : List of 2 floats, optional
        Bandpass filter executed by signaplot called inside function. The default is [0.02,0.2].
    freqlim : List of 2 floats, optional
        Frequency range in cycles per minute which will be plotted, passed to freqlim keyword in egg_signal_freq. The default is [1,10].
    vrange : TYPE, optional
        DESCRIPTION. The default is [0].
    figsize : Tuple, optional
        Figure size passed to figure creation. The default is (10,20).
    interpolation : TYPE, optional
        Interpolation passed to imshow. The default is 'bilinear'.
    n : uint, optional
        Overlap between segments, used for smoothness but slows down calcuation. The default is 10.
    intermediate : bool, optional
        Flag to generate intermediate plots (memory inefficient). The default is 'no'.
    max_scale : float, optional
        imshow max as a function of global max in the data. The default is .4.
    norm : bool, optional
        Flag to normalize frequency data between segments. The default is 'yes'.

    Returns
    -------
    None.

    '''
    lim_list=[xlim[0]]
    freq_array=[]
    while lim_list[-1]<xlim[1]:
        lim_list.append(lim_list[-1]+seg_length/n)# make a list of start points for the bins. Note that the last n of them are not useful because they would exit the overall limit
    for ele in lim_list[0:-1*n]:
        print([ele,ele+seg_length])
        #Run through Signalplot and then egg_signal_freq
        a,b,c=signalplot(dat,rate=rate,xlim=[ele,ele+seg_length],freq=freq,time=time)
        a1,b1,c1=egg_signalfreq(c,rate=rate,freqlim=freqlim,mode='power')
        
        if not intermediate:
            plt.close(a1)
            plt.close(a)
        ####
        
        
        ### Only select frequencies based on frequency limits in cycles per minute input
        loc=(c1[0,:]>freqlim[0]) & (c1[0,:]<freqlim[1])
        freq_dat=c1[1:9,loc]
        if norm:
            for i in range(0,8):
                freq_dat[i,:]=freq_dat[i,:]/freq_dat[i,:].sum()
        
        ####
        #print(freq_dat.shape)
        freq_array.append(freq_dat)
#        print(freq_dat.shape)
#    freq_array=np.array(freq_array)
#    freq_array[0]
    freq_data_list=[]
    for i in range(8):

        channel_freq=[]
        for j,ele in enumerate(freq_array):
            if len(channel_freq)==0: 
                channel_freq=ele[i,:]
            else:
                channel_freq=np.vstack((channel_freq,ele[i,:]))
        
#        print(channel_freq.shape)
        freq_data_list.append(channel_freq)
    
    freq_data_list=np.dstack(freq_data_list)
    fig_an, ax_an = plt.subplots(nrows=8, figsize=figsize)
    
    fig_an.supylabel('Frequency (cycles/min)')
    for i,ele in enumerate(ax_an):
        
        # Reminder for imshow you always have to flip to get it to fill from bottom to top
        # also reminder, matplotlib draws from top to bottom, which is why we start plotting with channel 7
        colors=ele.imshow(np.flip(freq_data_list[:,:,7-i].T,axis=0),aspect='auto',extent=[xlim[0],xlim[1],freqlim[0],freqlim[1]],cmap='jet',vmin=vrange[0],vmax=freq_data_list[:,:,7-i].max()*max_scale,interpolation='nearest',resample=True)
        cbar=fig_an.colorbar(colors,ax=ele)
        ele.set_ylabel("Chan " + str(7-i))
        if i==7:
            ele.set_xlabel('Time (s)')
        else:
            ele.tick_params(labelbottom=False) 
    # if len(vrange)==2:
    #     colors=ax_an.imshow(np.flip(freq_dat,axis=0),aspect='auto',extent=[xlim[0],xlim[1],-0.5,7.5],cmap='jet',vmin=vrange[0],vmax=vrange[1],interpolation=interpolation)
    # else:
    #     colors=ax_an.imshow(np.flip(freq_dat,axis=0),aspect='auto',extent=[xlim[0],xlim[1],-0.5,7.5],cmap='jet',interpolation=interpolation)  
    return fig_an,ax_an,freq_data_list
#%%
def egg_freq_heatplot_v2(dat, rate=62.5, xlim=[0,10000],seg_length=500,freq=[0.02,0.2],order=3,freqlim=[1,10],vrange=[0],figsize=(10,20), textsize=16,interpolation='bilinear',n=10,mmc=False,intermediate=False,max_scale=.4,norm=True,time='Synctime',skip_chan=[]):
    '''

    Parameters
    ----------
    dat : Pandas Dataframe
        Dataframe producted from read_egg_v3.
    rate : float
        The sampling rate in samples/seconds. The default is 62.5.
    xlim : list of 2 floats, optional
        Range in seconds that will be plotted in time. The default is [0,10000].
    seg_length : float, optional
        Time in seconds that will be used to calculate frequency. The default is 500.
    freq : List of 2 floats, optional
        Bandpass filter executed by signaplot called inside function. The default is [0.02,0.2].
    freqlim : List of 2 floats, optional
        Frequency range in cycles per minute which will be plotted, passed to freqlim keyword in egg_signal_freq. The default is [1,10].
    vrange : TYPE, optional
        DESCRIPTION. The default is [0].
    figsize : Tuple, optional
        Figure size passed to figure creation. The default is (10,20).
    interpolation : TYPE, optional
        Interpolation passed to imshow. The default is 'bilinear'.
    n : uint, optional
        Overlap between segments, used for smoothness but slows down calcuation. The default is 10.
    intermediate : bool, optional
        Flag to generate intermediate plots (memory inefficient). The default is 'no'.
    max_scale : float, optional
        imshow max as a function of global max in the data. The default is .4.
    norm : bool, optional
        Flag to normalize frequency data between segments. The default is 'yes'.

    Returns
    -------
    None.

    '''
    plt.rcParams['font.size'] = textsize
    lim_list = [xlim[0]]
    freq_array = []
    skip_chan_text = list(map(lambda x: "Channel "+str(x), skip_chan))
    labels = [x for x in dat.columns if x.startswith('Channel') and x not in skip_chan and x not in skip_chan_text]
    while lim_list[-1] < xlim[1]:
        lim_list.append(lim_list[-1] + seg_length/n)

    for ele in lim_list[0:-1*n]:
        print([ele, ele+seg_length])
        a, b, c = signalplot(dat, rate=rate, xlim=[ele, ele+seg_length], freq=freq, order=order, time=time, output='PD', labels=labels, skip_chan=skip_chan)
        a1, b1, c1 = egg_signalfreq(c, rate=rate, freqlim=freqlim, mode='power', mmc=mmc)
        
        if not intermediate:
            plt.close(a1)
            plt.close(a)

        loc = (c1[0, :] > freqlim[0]) & (c1[0, :] < freqlim[1])
        freq_dat = c1[1:, loc]
        if norm:
            for i in range(0, len(freq_dat)):
                freq_dat[i, :] = freq_dat[i, :] / freq_dat[i, :].sum()     
        freq_array.append(freq_dat)

    freq_data_list = []
    for i in range(len(labels)):
        channel_freq = []
        for j, ele in enumerate(freq_array):
            if len(channel_freq) == 0: 
                channel_freq = ele[i, :]
            else:
                channel_freq = np.vstack((channel_freq, ele[i, :]))
        freq_data_list.append(channel_freq)
    freq_data_list = np.dstack(freq_data_list)
    
    # Adjust for single channel case
    fig_an, ax_an = plt.subplots(nrows=max(1, len(labels)), figsize=figsize)
    single_channel = len(labels) == 1
    if single_channel:
        ax_an = [ax_an]  # Make ax_an iterable
    fig_an.supylabel('Frequency (cycles/min)')
    for i, ele in enumerate(ax_an):
        colors = ele.imshow(np.flip(freq_data_list[:, :, len(labels)-1-i].T, axis=0), aspect='auto', extent=[xlim[0], xlim[1], freqlim[0], freqlim[1]], cmap='jet', vmin=vrange[0], vmax=freq_data_list[:, :, len(labels)-1-i].max()*max_scale, interpolation='nearest', resample=True)
        cbar = fig_an.colorbar(colors, ax=ele)
        ele.set_ylabel(labels[len(labels)-1-i])
        if i == len(labels)-1:
            ele.set_xlabel('Time (s)')
        else:
            ele.tick_params(labelbottom=False)
    # if len(vrange)==2:
    #     colors=ax_an.imshow(np.flip(freq_dat,axis=0),aspect='auto',extent=[xlim[0],xlim[1],-0.5,7.5],cmap='jet',vmin=vrange[0],vmax=vrange[1],interpolation=interpolation)
    # else:
    #     colors=ax_an.imshow(np.flip(freq_dat,axis=0),aspect='auto',extent=[xlim[0],xlim[1],-0.5,7.5],cmap='jet',interpolation=interpolation)  
    return fig_an,ax_an,freq_data_list

#%%
def egg_signal_check(data,rate=62.5, xpoint=1000, slow_window=200, res_window=15, ecg_window=8,chan_select=0, close=True, s_freq=[0.02,0.25],r_freq=[.25,5],e_freq=[5,1000],figsize=(10,15),s_flim=[1,10],r_flim=[10,40],e_flim=[50,150],rncomb=0):
    
    a0,b0,c0=signalplot(data,rate=rate,xlim=[xpoint,xpoint+slow_window],freq=[0.001,1000])



    a1,b1,c1=signalplot(data,rate=rate,xlim=[xpoint,xpoint+slow_window],freq=s_freq)
    aa1,bb1,cc1=egg_signalfreq(c1,rate=rate,freqlim=s_flim,mode='power',clip=True)
    maxloc=cc1[chan_select+1,:].argmax()
    s_peakfreq=cc1[0,maxloc]
    print('Peak Slow Wave Frequency is ', s_peakfreq)
    
    a2,b2,c2=signalplot(data,rate=rate,xlim=[xpoint,xpoint+res_window],freq=r_freq)
    aa2,bb2,cc2=egg_signalfreq(c2,rate=rate,freqlim=r_flim,mode='power',clip=True)
    maxloc=cc2[chan_select+1,:].argmax()
    res_peakfreq=cc2[0,maxloc]
    print('Peak Respiration Frequency is ', res_peakfreq)

    
    if rncomb==0: rncomb=res_peakfreq/60
    a3,b2,c3=signalplot(data,rate=rate,xlim=[xpoint,xpoint+ecg_window],freq=e_freq,ncomb=rncomb)
    aa3,bb3,cc3=egg_signalfreq(c3,rate=rate,freqlim=e_flim,mode='power',clip=True)
    maxloc=cc3[chan_select+1,:].argmax()
    e_peakfreq=cc3[0,maxloc]
    print('Peak ECG Frequency is ', e_peakfreq)


    cc1=cc1.T
    cc2=cc2.T
    cc3=cc3.T
    
    if close:
        plt.close(a0)
        plt.close(a1)
        plt.close(a2)
        plt.close(a3)
    fig,ax_n=plt.subplots(nrows=4,figsize=figsize)
    ax_n[0].plot(c0[:,0],c0[:,chan_select+1])
    ax_n[0].set_ylabel('Raw Data (mV)')
    
    ax_n[1].plot(c1[:,0],c1[:,chan_select+1])
    ax_n[1].set_ylabel('Slow Wave (mV)')
    
    ax_n[2].plot(c2[:,0],c2[:,chan_select+1])
    ax_n[2].set_ylabel('Respiration (mV)')
    
    ax_n[3].plot(c3[:,0],c3[:,chan_select+1])
    ax_n[3].set_ylabel('ECG (mV)')
    ax_n[3].set_xlabel('Time (s)')
    
    fig2,ax_n2=plt.subplots(nrows=3,figsize=figsize)

    ax_n2[0].stem(cc1[:,0],cc1[:,chan_select+1])
    ax_n2[0].set_ylabel('Slow Wave Power (dB)')
    
    ax_n2[1].stem(cc2[:,0],cc2[:,chan_select+1])
    ax_n2[1].set_ylabel('Respiration Power (dB)')
    
    ax_n2[2].stem(cc3[:,0],cc3[:,chan_select+1])
    ax_n2[2].set_ylabel('ECG Power (dB)')
    ax_n2[2].set_xlabel('Frequency (Hz)')
    
    return fig,ax_n    

#########PEAK TRACKING LIBRARIES BETA

def time_and_HR (array_filepath, seg_length = 30, time='mean', plot='yes',peak='yes', thres=0.5):
    """
    Function to plot the Time vs. HR from data in  exported by signal plot (2 column)
    
    Inputs:
        array_filepath: import your array.txt in here
        seg_length: The number of seconds you want each peak detection graph to be
        time: 3 options: 'max, 'min', 'mean'. 'max' sets the largest time value in each array for each HR, 'min' takes the smallest, and 'mean' takes the average. 
        plot: option to plot time vs. HR graph. plot='yes' to plot, plot= anything else to not plot
        peak: option to plot the peak graphs. plot='yes' to plot, plot= anything else to not plot
        thres:(float between [0., 1.]) – Normalized threshold. Only the peaks with amplitude higher than the threshold will be detected.
        
    Outputs:
        blank_array: an array with all data values of Time vs. HR
    """
    array= np.genfromtxt(array_filepath)
    x_axis=array[:,0]
    y_axis=array[:,1]*-1
    num_of_xsplits=round((x_axis.max()-x_axis.min())/seg_length) #formula for splitting entire time into 20 segments of same length
    
    splitx= np.array_split(x_axis, num_of_xsplits)  #actually splitting up the time
    splity= np.array_split(y_axis, num_of_xsplits) # splitting up the voltage as well
    
    
    blank_array= np.zeros([num_of_xsplits,2], dtype=float)  #creating an array of zeros on 2 columns
    for i,arr in enumerate(splitx):
        new_indexes = peakutils.indexes(splity[i], thres, min_dist=20) #using peak_detection library to find moments of peaks
        total_time = splitx[i].max()-splitx[i].min()    
        HR = 60*len(new_indexes)/total_time                #calculating HR by dividing total number of peaks by total time
        blank_array[i,1]=HR
        if time=='mean': blank_array[i,0]=arr.mean()
        if time=='min' : blank_array[i,0]=arr.min()
        if time=='max' : blank_array[i,0]=arr.max()
        if peak=='yes':### I put this into the for loop so it plots all of the segments - Sean
            pyplot.figure(figsize=(10,6))
            pplot(splitx[i], splity[i], new_indexes)
    if plot=='yes':
        fig=pyplot.figure()
        ax=fig.add_subplot(111)        
        ax.plot(blank_array[:,0],blank_array[:,1])
    print('++++++++++++++++++++++++++++')
    print(blank_array)
    return blank_array




def import_PR (CSV_filepath, droplist, offset=0):
    """
    Function to plot the Time vs. PR from data in CSV file generated by animal monitor
    
    Inputs:
        CSV_filepath: filepath of CSV file
        droplist: list of rows that you want to drop.
        offset: the amount of time offset to match the time_and_HR graph
        
    Outputs:
        blank_array: an array with all data values of Time vs. PR
    """
    df= pd.read_csv(CSV_filepath, skiprows=3,usecols=[0,1,2,3,4]) #skipping some rows and using only first 4 columns
    df2=df.drop(droplist)                                #getting rid of data to match starting point w/ HR data
    df3=df2.drop(df2[df2['PR']=='--'].index) 
        
    index =range(len(df3)) # creating a new index range bc of all the df.drops
    df3.index = index
    df3['Time']=df3['Time'].str[:-3] #getting rid of the :00 seconds from all times to make easier calculations
    hours_to_sec=df3['Time'].str.split(':', n=1).str[0].astype('float')*3600 #splitting up the times by ':' and then turning hours into seconds
    min_to_sec= df3['Time'].str.split(':', n=1).str[1].astype('float')*60 #splitting up times and then turning minutes into seconds
    total_sec= (hours_to_sec + min_to_sec)-offset #adding seconds together and offsetting to match w/ HR datapoints       
    x=total_sec
    y=df3['PR'].astype("float")
    
    array=np.column_stack((x,y))  #combines both columns
    return array
    



def plot_peak_freq(filepath,seg_length, save_fig_location='', thres=.5,xlim=[0,100000],freq=[.02,.2],min_dist=20):
    """
    Function to plot graph of all channels frequencies
    
    Inputs:
        CSV_filepath: filepath of CSV file
        seg_length: the size of the pieces of data that the peak detection will run on, in seconds. 
        save_fig_location: filepath to store all the peak detection graphs
        thres:(float between [0., 1.]) – Normalized threshold. Only the peaks with amplitude higher than the threshold will be detected.
        xlim: list of 2 elements taking time range of interest. Default behavior is to full timescale
        freq: frequency list in Hz, 2 element list, for bandpass filtering. Default is no filtering
        min_dist: (int) Minimum distance in number of array elements between each detected peak. The peak with the highest amplitude is preferred to satisfy this constraint.
    Outputs:
        output_arr: array of time vs. all channels' HR
        axs: plots frequencies of all channels in separately
        ax: plots frequenices of all channels in single graph
    """
    dat= read_egg_v3(filepath,header=0,rate=62.5,scale=150,error=0)
    x=signalplot(dat,xlim=xlim,spacer=0,vline=[],freq=freq,order=3,rate=62.5, title='',skip_chan=[],figsize=(10,20),textsize=16,hline=[])
    array_shape=x[2].shape
    fig=plt.figure(figsize=(10,6))                            #graphs of frequencies
    ax=fig.add_subplot(111)
    fig, axs= plt.subplots(nrows=8, ncols=1,figsize=(10,20),sharex=True,sharey=True)
    output_arr=np.array([])
    
    for i in range(1,array_shape[1]):
        x_axis=x[2][:,0]
        y_axis=x[2][:,i]*-1

        
        num_of_xsplits=round((x_axis.max()-x_axis.min())/seg_length) 
        splitx= np.array_split(x_axis, num_of_xsplits)  #actually splitting up the time
        splity= np.array_split(y_axis, num_of_xsplits) # splitting up the voltage as well

         
        
        blank_array= np.zeros([num_of_xsplits,2], dtype=float)  #creating an array of zeros on 2 columns
        for j,arr in enumerate(splitx):
            new_indexes = peakutils.indexes(splity[j], thres=thres, min_dist=min_dist) #using peak_detection library to find moments of peaks
            total_time = splitx[j].max()-splitx[j].min()    
            HR = 60*len(new_indexes)/total_time                #calculating HR by dividing total number of peaks by total time
            blank_array[j,0]=arr.mean()
            blank_array[j,1]=HR
            if save_fig_location!='':
                fig_temp=pyplot.figure(figsize=(10,6))
                print(i,j)
                print(new_indexes)
            
                if len(new_indexes)>0: 
                    pplot(splitx[j], splity[j], new_indexes)                               #peak graph
                    fig_temp.savefig(save_fig_location+'Channel'+str(i)+"_Segment_"+str(j)+".png")
                plt.close(fig_temp)
                
        if len(output_arr)==0: 
            output_arr=blank_array[:,0]
        output_arr=np.column_stack((output_arr,blank_array[:,1]))
              
        ax.plot(blank_array[:,0],blank_array[:,1],label='Channel ' + str(i))
        axs[i-1].plot(blank_array[:,0],blank_array[:,1])
        
        
    ax.legend(fontsize=10)
    return output_arr, axs, ax

def power_vs_time(arr):
    time=arr[:,0].max()-arr[:,0].min()
    columns=len(arr[0,:])-1
    output=np.zeros(columns)
    for i in range(1,columns+1):
        channel = arr[:,i]
        output[i-1]=(np.abs(channel)**2).sum()/len(arr[:,0])
    return output


def signal_peakcalculate(c,channel=0,plot=True,width=10,invert=False,trim=[0,1],threshold=0.02,distance=10):
    i=channel+1 #move index up by 1 as time is first column
    c=c[trim[0]:,:]
    c=c[:-trim[1],:]
    d=c[:,i]
    t=c[:,0]
    if invert: d=-1*d
    d=d-d.mean() #zero out to baseline
    if width==1:
        peaks,b=sig.find_peaks(d,threshold=threshold,distance=distance)
    else:    
        peaks=sig.find_peaks_cwt(d,widths=width)
    if plot:
        fig,ax=plt.subplots()
        ax.plot(t,d)
        ax.plot(t[peaks],d[peaks],'ro')
    print('Mean',d[peaks].mean())
    print('STD',d[peaks].std())
    return t[peaks],d[peaks]
        
# %%
