'''
Comparison of two models for the same data
------------------------------------------

    In this example, two models (exponential and linear) are fitted
to data from a single Dataset.
'''

###########
# Imports #
###########

# import everything we need from kafe
import kafe
# script test_readColumnData.py
# -*- coding: utf-8 -*-

import sys, numpy as np, matplotlib.pyplot as plt
from PhyPraKit import odFit, labxParser, readColumnData
# additionally, import the two model functions we
# want to fit:
from kafe.function_library import linear_2par, exp_2par


############ def Mittelwert
N=3. #da drei werte pro Messung
def mean(a):
  return np.sum(a)/N

def variance(a):
    v=np.sum((a-mean(a))**2)/(N-1.)
    return v

def sigma(a):
    s= np.sqrt(variance(a))
    return s

def sigmamean(a):
    return sigma(a)/np.sqrt(N)

###############################
#Einlesen Daten


fname='Aufg1_2.dat'
ncols=7
data_array, info_dict =\
  readColumnData(fname, ncols, delimiter=' ', pr=False)

# print what we got:
Abstand=data_array[:,0] # 1st column
meinso  =data_array[:,1]/5. # 2nd column
meinsu  =data_array[:,2]/5.
mzweio  =data_array[:,3]/5.
mzweiu  =data_array[:,4]/5.
mdreio  =data_array[:,5]/5.
mdreiu  =data_array[:,6]/5.

Abstand=Abstand/100.

##Mittelwert bilden
mao=np.zeros(N)
mau=np.zeros(N)
mmalleo =np.zeros(len(Abstand))
mmalleu =np.zeros(len(Abstand))
fmmalleo=np.zeros(len(Abstand))
fmmalleu=np.zeros(len(Abstand))

for i in range(len(Abstand)):
    mao=(meinso[i],mzweio[i],mdreio[i])
    mau=(meinsu[i],mzweiu[i],mdreiu[i])
    mmalleo[i] =mean(mao)
    mmalleu[i] =mean(mau)
    fmmalleo[i]=sigmamean(mao)
    fmmalleu[i]=sigmamean(mau)

xdatao=Abstand
ydatao=mmalleo
yerroro=fmmalleo

# fur den ersten Plot
untenplot=mmalleu

#fuer untere schneide nicht lineare punkte rausnehmen
n=6 #gefunde lineare Punkte
xdatau=np.zeros(n)
ydatau=np.zeros(n)
yerroru=np.zeros(n)
for k in range(n):
    xdatau[k]=Abstand[k+2]
    ydatau[k]=mmalleu[k+2]
    yerroru[k]=fmmalleu[k+2]
    
    
print xdatau
####################
# Helper functions #
####################

def generate_datasets(output_file_path1,output_file_path2):
    '''The following block generates the Datasets and writes a file for
    each of them.'''

    import numpy as np  # need some functions from numpy
    
    my_datasets = []
    my_datasets.append(kafe.Dataset(data=(xdatao, ydatao)))
    my_datasets[-1].add_error_source('y', 'simple', yerroro)
    
    my_datasets.append(kafe.Dataset(data=(xdatau, ydatau)))
    my_datasets[-1].add_error_source('y', 'simple', yerroru)
    
    my_datasets[0].write_formatted(output_file_path1)
    my_datasets[1].write_formatted(output_file_path2)

############
# Workflow #
############

# Generate the Dataset and store it in a file
generate_datasets('Aufg1_2obereSchn.dat',
                  'Aufg1_2untenSchn.dat')

# Initialize Dataset
my_datasets = [kafe.Dataset(title="Periodendauer obere Schneide"),
               kafe.Dataset(title="Periodendauer untere Schneide")]
# Load the Dataset from the file
my_datasets[0].read_from_file(input_file='Aufg1_2obereSchn.dat')
my_datasets[1].read_from_file(input_file='Aufg1_2untenSchn.dat')

# Create the Fits

my_fits = [kafe.Fit(dataset,linear_2par,
                    fit_label="Linear regression ")
           for dataset in my_datasets]
# Do the Fits
for fit in my_fits:
    fit.do_fit()

# Create the plots
my_plot = kafe.Plot(my_fits[0],my_fits[1])#,my_fits[1])



#Verschoenern
my_plot.axis_labels = ['$Periodenanzahl$', '$Zeit$ (s)']

# Draw the plots
my_plot.plot_all()  # only show data once (it's the same data)
my_plot.save('Aufg1_2kafe.pdf')

fig=plt.figure()
#ax1=fig.add_subplot(1,1,1)

plt.plot(Abstand,mmalleo, 'rs',linewidth=1, label='obere Schneide')
plt.plot(Abstand,untenplot, 'bs', label='untere Schneide')
plt.axis([0.59,0.69,1.57,1.65])
ax1.set_xlabel('$Abstand$ (m)',size='large')
ax1.set_ylabel('$Periodendauer$ T (s)',size='large')

#plt.savefig('Aufg1_2plot.pdf')


plt.show()
