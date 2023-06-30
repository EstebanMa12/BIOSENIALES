## -*- coding: utf-8 -*-
#
#
## libreria para hacer graficos tipos matlab (pyplot)
#import matplotlib.pyplot as plt;
##libreria de manejo de arreglos de grandes dimensiones (a diferencia de las listas basicas de python)
#import numpy as np;
##libreria con rutinas de PDS
#import scipy.signal as signal;


#%%
import numpy as np;
import matplotlib.pyplot as plt
from csv import reader as reader_csv;
import scipy.signal as signal;

#absolute route
#data=open('G:\\Mi unidad\\Laboratorio_Neurofisiologia\\proyectos_convocatorias\\anesthesia\\registros\\15-11-2018\\P1_RAWEEG_2018-11-15_Ensayo_1min.txt',"r");
#relative route, in this case the file to be open must be in the same location
#as the python file
data=open('P1_RAWEEG_2018-11-15_Ensayo_1min.txt',"r");
lines = reader_csv(data);

row_number = 0;
header = '';
channels = 11; # 8 EEG and 3 accelerometer
header_size = 6; #number of rows before the EEG data

#list to save the loaded data
data = [];

for row in lines:
    #reading header, only printing the content
    if row_number < header_size:
        header = header + row[0] + '\n';
        row_number = row_number + 1;
        print(row);
    else: #reading data
        #the data must be channels by times
        temp = []; #store the data
        counter = 0; #column counter
        #for each value in the row
        for column in row:
            #the first value is an index and the last is a time: skip
            if counter == 0:
                counter = counter + 1;
                continue;
            elif counter == channels + 1:
                break;
            else:
                temp.append(float(column));
            
            counter = counter + 1;
        data.append(temp);
    
#transform the list to ndarray    
biosignal = np.asarray(data, order = 'C');
#transpose to have channels as rows and samples as columns
biosignal = np.transpose(biosignal);

plt.plot(biosignal[1,1000:]);
plt.show();

time = 1*60;
fs = 250;

# biosignalf = eegfiltnew(biosignal, fs, 1, 0);
# #biosignalf = eegfiltnew(biosignalf, fs, 0, 30);

# biosignalf = biosignalf[0:8,fs*time*2:fs*time*3];
# biosignal = biosignal[0:8,fs*time*2:fs*time*3];

# plt.subplot(2,1,1);
# plt.plot(biosignalf[4,:]);
# plt.subplot(2,1,2);
# plt.plot(biosignal[4,:]);
# plt.show();

#%%

#PRIMERA PARTE CARGA Y MANIPULACION BASICA
#library to load mat files
import scipy.io as sio;

#loading data
mat_contents = sio.loadmat('C001R_EP_reposo.mat')
#the data is loaded as a Python dictionary
print("the loaded keys are: " + str(mat_contents.keys()));
#in the current case the signal is stored in the data field
data = mat_contents['data'];

print("Variable python: " + str(type(data)));
print("Tipo de variable cargada: " + str(data.dtype));
print("Dimensiones de los datos cargados: " + str(data.shape));
print("Numero de dimensiones: " + str(data.ndim));
print("Tamanio: " + str(data.size));
print("Tamanio en memoria (bytes): " + str(data.nbytes));

#%%

# la frecuencia de muestreo ya es conocida
fs = 1000;

#Se extrae el canal 1, todos los puntos, todas las epocas
channel_1 = data[0,:,:];
print(channel_1.shape);

#se recupera el numero de puntos y epoca en los datos
(points, epochs) = channel_1.shape;

#se elimina la estructura de las epocas 
channel_1 = np.reshape(channel_1, (points*epochs),order='F');
print(channel_1.shape);
print(data.shape);

#algunas veces es necesario eliminar las dimensiones vacias
channel_1 = np.squeeze(channel_1);
print(channel_1.shape);

#SEGUNDA PARTE GRAFICACION

#se crea una copia
data_temp = data.copy();

#esta asignacion no crea una copia
channel_1 = data[0,:,:];
channel_1[:,:] = 0;

plt.figure();
plt.plot(channel_1);
plt.title('Primer grafico');

#se ha modificado la variable data
plt.figure();
plt.plot(data[0,:,:]);
plt.title('Alteracion de la variable origina!!!');

#make data by epochs
channel_1 = np.reshape(channel_1, (points,epochs));
print(channel_1.shape);

#extract the first epoch
channel_1 = channel_1[:,1];
shape_channel_1 = channel_1.shape;
print(channel_1.shape);

#TERCERA PARTE ANALISIS ESTADISTICO BASICO

plt.figure();
plt.subplot(2,1,1);
plt.plot(channel_1)
f, Pxx = signal.welch(channel_1,fs,'hamming', 512, 256, 512, scaling='density');
plt.subplot(2,1,2);
plt.stem(f,Pxx);
plt.xlim([0, 100]);
plt.show();

#anadir tiempo
tiempo = np.linspace(0,shape_channel_1[0]/fs,int(((shape_channel_1[0]/fs) - 0.0)*fs),endpoint=False);

plt.figure();
plt.subplot(2,1,1);
plt.plot(tiempo,channel_1);

#%%

#PRIMERA PARTE CARGA Y MANIPULACION BASICA
#library to load mat files
import scipy.io as sio;

#loading data
mat_contents = sio.loadmat('senales_potencial.mat')
#the data is loaded as a Python dictionary
print("the loaded keys are: " + str(mat_contents.keys()));
#in the current case the signal is stored in the data field
condicion_frecuente = mat_contents['frecuente'];
condicion_infrecuente = mat_contents['infrecuente'];

print("Dimensiones de los datos cargados: " + str(condicion_frecuente.shape));
print("Dimensiones de los datos cargados: " + str(condicion_infrecuente.shape));


#%%
import pandas as pd

import scipy.io as sio;
import numpy as np;

#loading data
mat_contents = sio.loadmat('promedios.mat')
#the data is loaded as a Python dictionary
print("the loaded keys are: " + str(mat_contents.keys()));
#in the current case the signal is stored in the data field
promedio_frecuente = mat_contents['frecuente_promedio'];
promedio_infrecuente = mat_contents['infrecuente_promedio'];

#create a 'table' with the data of the averages
promedios = np.concatenate((promedio_frecuente, promedio_infrecuente));
#we change of 2x1150 to 1150x2
promedios = promedios.transpose();
#we can have as index the time of adquisition of the signal 150 ms before the
#stimulus and 1 second after the stimulus, acquiring a sample each 1 ms
indice_tiempo = pd.Index(np.arange(-0.150,1,1/1000));

#the DataFrame is created
potencial = pd.DataFrame(promedios,index=indice_tiempo,columns=('frecuente',
                                                                'infrecuente'));
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid")
plt.figure(figsize=(10,8))
ax = sns.boxplot(x='frecuente', data=potencial, orient="v")
                                                                



