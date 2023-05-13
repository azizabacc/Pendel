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


fname='Aufg2eich.dat'
ncols=4
data_array, info_dict =\
  readColumnData(fname, ncols, delimiter=' ', pr=False)

# print what we got:
nperiode=data_array[:,0] # 1st column
meins=data_array[:,1] # 2nd column
mzwei=data_array[:,2] # 3rd column
mdrei=data_array[:,3] # 4th column

##Mittelwert bilden
ma=np.zeros(N)
mmalle =np.zeros(len(meins))
fmmalle=np.zeros(len(meins))

for i in range(len(meins)):
    ma=(meins[i],mzwei[i],mdrei[i])
    mmalle[i] =mean(ma)
    fmmalle[i]=sigmamean(ma)

ydata=mmalle
xdata=nperiode
yerror=fmmalle

print fmmalle

####################
# Helper functions #
####################

def generate_datasets(output_file_path):
    '''The following block generates the Datasets and writes a file for
    each of them.'''

    import numpy as np  # need some functions from numpy
    
    my_datasets = []
    my_datasets.append(kafe.Dataset(data=(xdata, ydata)))
    my_datasets[-1].add_error_source('y', 'simple', yerror)
    
    my_datasets[0].write_formatted(output_file_path)


############
# Workflow #
############

# Generate the Dataset and store it in a file
generate_datasets('Aufg2eichkafe.dat')

# Initialize Dataset
my_datasets = [kafe.Dataset(title="Zeit")]
# Load the Dataset from the file
my_datasets[0].read_from_file(input_file='Aufg2eichkafe.dat')


# Create the Fits

my_fits = [kafe.Fit(dataset,linear_2par,
                    fit_label="Linear regression ")
           for dataset in my_datasets]
# Do the Fits
for fit in my_fits:
    fit.do_fit()

# Create the plots
my_plot = kafe.Plot(my_fits[0])#,my_fits[1])



#Verschoenern
my_plot.axis_labels = ['$Periodenanzahl$', '$Zeit$ (s)']

# Draw the plots
my_plot.plot_all()  # only show data once (it's the same data)

###############
# Plot output #
###############

# Save the plots
#my_plot.save('Aufg2eich.pdf')

'''
# check contours
contour1 = my_fits[0].plot_contour(0, 1, dchi2=[1.,2.3])
profile00=my_fits[0].plot_profile(0)
profile01=my_fits[0].plot_profile(1)
contour2 = my_fits[1].plot_contour(0, 1, dchi2=[1.,2.3])
'''
#contour1.savefig('kafe_example1_contour1.pdf')
#contour2.savefig('kafe_example1_contour2.pdf')
#profile00.savefig('kafe_example1_profile00.pdf')
#profile01.savefig('kafe_example1_profile01.pdf')

# Show the plots
my_plot.show()
