# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:36:32 2023

@author: seany
"""

import sys
sys.path.append('C:/Users/seany/Dropbox/Langer Lab/Data/EGG_Data_Repository/')
from Old_Plot_EGG import * 

#%%

path='2022.08.17_Full.txt'
path2='2022.05.02_Combined.txt'
data_long=read_egg_v3(path,scale=600)
data=read_egg_v3(path2,scale=600)


#%%

#a,b,c1=signalplot(data_long,freq=[.0005,15])
a1,b1,c1=signalplot(data_long,freq=[.0001,0.01])


#a,b,c=signalplot(data,freq=[0.0005,.01],xlim=[500,5500])

#egg_signalfreq(c,rate=62.5,freqlim=[0.05,.6],ylim=0,mode='power',log=False,clip=False)
d1,e1,f1= egg_signalfreq(c1,rate=62.5,freqlim=[0.08,.6],ylim=0,mode='power',xlog=False,clip=False)


a2,b2,c2=signalplot(data_long,freq=[.01,0.25])
d2,e2,f2=egg_signalfreq(c2,rate=62.5,freqlim=[2,10],ylim=0,mode='power',xlog=False,clip=False)


#%%

#pathday0='20230419_Day0_data_laptop.txt'
#pathday1='20230412_Day1_data.txt'
#pathday2='20230412_Day2_data.txt'
path='2023.04.19_AnimalPC01/2023.04.19_AfterPlacement.txt'
data_day0=read_egg_v3(path,scale=600)

path1='20230420_day1_data_laptop.txt'
data_day1=read_egg_v3(path1,scale=600)