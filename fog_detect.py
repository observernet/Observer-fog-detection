#!/usr/bin/env python
# coding: utf-8
# 
import pandas as pd
import numpy as np
import os
import joblib
from keras.models import load_model
import datetime
import psycopg2

# data load
path = './fog/data/'
files = sorted([i for i in os.listdir(path) if i[-3:]=='raw'])
print('--------file name:',path+files[-1],'----------')

f = open(path+files[-1])
raw = f.readlines()
f.close()
data = [[files[-1][8:10]]+datum.strip('\n').split(';')[:-1] for datum in raw[-600:]]
data = pd.DataFrame(data)

MMScaler = joblib.load('./fog/model/MMScaler.pkl')

# preprocess

col=['mon','기기고유번호','위도','경도','시간','PRESS','기압QC','T','기온QC','RH',
     '습도QC','PTY','강수QC','PM10','PM10 QC','PM25','PM25_R','PM25 QC']
DATA = pd.DataFrame(data)
DATA.columns = col
DATA = DATA[['기기고유번호','위도','경도','시간','PRESS','T','RH','PTY','PM10','PM25','PM25_R']]
DATA = DATA.replace('-999',np.nan)

def vs_class(x):
    if 4800<x:
        return 0
    elif 2000<x and x<=4800:
        return 1
    elif x<=2000:
        return 2

X = []
station_id=[]
Rain = []
RH = []
XDF = pd.DataFrame(columns=DATA.columns)

for i in set(DATA['기기고유번호']):
    tmp = DATA[DATA['기기고유번호']==i].astype('float')
    tmp['PM25_MA'] = tmp['PM25'].rolling(10).mean()
    Rain.append(tmp['PTY'].values[-1])
    RH.append(tmp['RH'].values[-1])
    XDF = XDF.append(tmp.iloc[-1])
    tmp = tmp[['PRESS', 'T','RH', 'PTY', 'PM25_MA', 'PM25','PM10']].iloc[-10:]
    X.append(MMScaler.transform(tmp.values))
    station_id.append(i)


# model load
model = load_model('/fog/model/fog_detect.h5')

# inference
predclass = [vs_class(model.predict(np.array(i).reshape(-1,10,7))) for i in X]
pred = [model.predict(np.array(i).reshape(-1,10,7))[0][0] for i in X]

fgd_result_data = pd.DataFrame({'station_id':station_id,
                                'created_timestamp':len(station_id)*[datetime.datetime.now()],
                                'detected_fog':predclass,
                                'detected_fog_pre':predclass,
                                'vs': pred})
fgd_result_data['created_timestamp'] = fgd_result_data['created_timestamp'].apply(lambda X: 
                                                                    str(X)[:16].replace(':','').replace('-','').replace(' ',''))

# post-process

for i in range(len(fgd_result_data)):
    mon = int(fgd_result_data['created_timestamp'].iloc[i][4:6])
    if Rain[i] < 0:  
        fgd_result_data['detected_fog'].iloc[i] = 3
    elif RH[i] <= 58:
        fgd_result_data['detected_fog'].iloc[i] = 0
    else:
        if mon==6 or mon==7 or mon==8 or mon==9:
            if RH[i]<=80 and fgd_result_data['detected_fog'].iloc[i]==2:
                fgd_result_data['detected_fog'].iloc[i] = 1
        else:
            if RH[i]<=85 and fgd_result_data['detected_fog'].iloc[i]==2:
                fgd_result_data['detected_fog'].iloc[i] = 1

# result
XDF['기기고유번호'] = XDF['기기고유번호'].astype('int').astype('str')
sDF = pd.merge(fgd_result_data,XDF, left_on='station_id', right_on='기기고유번호', how='outer')
sDF = sDF[['station_id','created_timestamp','detected_fog','detected_fog_pre',
           'vs','PRESS','T','RH','PTY','PM10','PM25','PM25_R','PM25_MA']]

relist = os.listdir('./fog/output')
now = str(datetime.datetime.now())
title = now[:4]+now[5:7]+now[8:10]
if 'fog_DnF_'+title+'.txt' in relist:
    sDF.to_csv('./fog/output/fog_DnF_'+title+'.txt',index=False, header=False, mode='a')
else:
    sDF.to_csv('/fog/output/fog_DnF_'+title+'.txt',index=False)

fgd_result_data = fgd_result_data.fillna(0)


print('-----------Done---------------')
