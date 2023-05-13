'''
A Tale of Two Fits
------------------

    This simple example demonstrates the fitting of a linear function to
    two Datasets and plots both Fits into the same Plot.
'''

###########
# Imports #
###########

# import everything we need from kafe
import kafe
from kafe import ASCII, LaTeX, FitFunction
# additionally, import the model function we
# want to fit:
from kafe.function_library import linear_2par
import numpy as np


####################
# Helper functions #
####################

# -- define test configuration 



# def Mittelwert
N=3 #da drei werte pro Messung
def mean(a):
  return np.sum(a)/N

def variance(a):
    v=np.sum((a-mean(a))**2)/(N-1)
    return v

def sigma(a):
    s= np.sqrt(variance(a))
    return s

def sigmamean(a):
    return sigma(a)/np.sqrt(N)



PVC=[(0.32, 0.35, 0.34),(0.8,0.7,0.7),(1.82, 1.83, 1.82),(3.71,3.63,3.59)]
MES=[(0.03,0.03,0.0300001),(0.07,0.07,0.06),(0.17,0.17,0.16),(0.41,0.39,0.39)]
CU =[(0.02,0.01,0.02),(0.05,0.05,0.06),(0.16,0.16,0.15),(0.33,0.33,0.32)]
ALU=[(0.07,0.07,0.06),(0.13,0.13,0.14),(0.32,0.32,0.33),(0.63,0.6300001,0.63)]
FE =[(0.015,0.0125,0.012),(0.04,0.0400001,0.04),(0.10,0.100001,0.10),(0.22,0.21,0.21)]

i=0

meanPVC=np.zeros(4)
meanMES=np.zeros(4)
meanCU =np.zeros(4)
meanALU=np.zeros(4)
meanFE =np.zeros(4)
sigmamPVC=np.zeros(4)
sigmamMES=np.zeros(4)
sigmamCU =np.zeros(4)
sigmamALU=np.zeros(4)
sigmamFE =np.zeros(4)
while i<4:
    meanPVC[i]=mean(PVC[i])
    meanMES[i]=mean(MES[i])
    meanCU [i]=mean(CU [i])
    meanALU[i]=mean(ALU[i])
    meanFE [i]=mean(FE [i])
    sigmamPVC[i]=sigmamean(PVC[i])
    sigmamMES[i]=sigmamean(MES[i])
    sigmamCU [i]=sigmamean(CU[i])
    sigmamALU[i]=sigmamean(ALU[i])
    sigmamFE [i]=sigmamean(FE[i])
    i=i+1

m=(0.1,0.2,0.5,1.)
# Kraft abhaengigkeit mit Messuhr   /1000 wegen g zu kg
def F(masse,s):
    return (m-3.6*s/1000.)*9.81


xdataPVC=F(m,meanPVC)
xdataMES=F(m,meanMES)
xdataCU =F(m,meanCU )
xdataALU=F(m,meanALU)
xdataFE =F(m,meanFE )

ydataPVC=meanPVC
ydataMES=meanMES
ydataCU =meanCU 
ydataALU=meanALU
ydataFE =meanFE


def generate_datasets(output_file_path1, output_file_path2, output_file_path3, output_file_path4, output_file_path5):
    '''The following block generates the Datasets and writes a file for
    each of them.'''

    import numpy as np  # need some functions from numpy

    my_datasets = []


    my_datasets.append(kafe.Dataset(data=(xdataPVC, ydataPVC)))
    my_datasets[-1].add_error_source('x', 'simple', sigmamPVC/10.)
    my_datasets[-1].add_error_source('y', 'simple', sigmamPVC)


    my_datasets.append(kafe.Dataset(data=(xdataMES, ydataMES)))
    my_datasets[-1].add_error_source('x', 'simple', sigmamMES/10.)
    my_datasets[-1].add_error_source('y', 'simple', sigmamMES)
    
    my_datasets.append(kafe.Dataset(data=(xdataCU , ydataCU )))
    my_datasets[-1].add_error_source('x', 'simple', sigmamCU/10. )
    my_datasets[-1].add_error_source('y', 'simple', sigmamCU)
    
    my_datasets.append(kafe.Dataset(data=(xdataALU, ydataALU)))
    my_datasets[-1].add_error_source('x', 'simple', sigmamALU/10.)
    my_datasets[-1].add_error_source('y', 'simple', sigmamALU)
    
    my_datasets.append(kafe.Dataset(data=(xdataFE , ydataFE )))
    my_datasets[-1].add_error_source('x', 'simple', sigmamFE/10.)
    my_datasets[-1].add_error_source('y', 'simple', sigmamFE)

    my_datasets[0].write_formatted(output_file_path1)
    my_datasets[1].write_formatted(output_file_path2)
    my_datasets[2].write_formatted(output_file_path3)
    my_datasets[3].write_formatted(output_file_path4)
    my_datasets[4].write_formatted(output_file_path5)
    

############
# Workflow #
############


# Generate the Dataseta and store them in files

'''
generate_datasets('datasetPVC2.dat', 
                  'datasetMES2.dat',
                  'datasetCU_2.dat',
                  'datasetALU2.dat',
                  'datasetFE_2.dat')
'''
# Initialize the Datasets
my_datasets = [kafe.Dataset(title="PVC"),
               kafe.Dataset(title="Messing"),
               kafe.Dataset(title="Kupfer"),
               kafe.Dataset(title="Aluminium"),
               kafe.Dataset(title="Eisen")]

# Load the Datasets from files
my_datasets[0].read_from_file(input_file='datasetPVC.dat')
my_datasets[1].read_from_file(input_file='datasetMES.dat')
my_datasets[2].read_from_file(input_file='datasetCU_.dat')
my_datasets[3].read_from_file(input_file='datasetALU.dat')
my_datasets[4].read_from_file(input_file='datasetFE_.dat')



# Create the Fits
my_fits = [kafe.Fit(dataset,
                    linear_2par,
                    fit_label="Linear regression " )
           for dataset in my_datasets]

# Do the Fits
for fit in my_fits:
    fit.do_fit(quiet=False, verbose=True)

# Create the plots
my_plot = kafe.Plot(my_fits[0],
                    my_fits[1],
                    my_fits[2],
                    my_fits[3],
                    my_fits[4])

#Verschoenern
my_plot.axis_labels = ['$F \ in \ N$', '$s \ in \ mm$']


#kafe.draw_fit_parameters_box(plot_spec=0, force_show_uncertainties=False)
#kafe.fit.round_to_significance(value, error, significance=2)



#my_plot.axis_units  = ['N', 'mm']
# Draw the plots
my_plot.plot_all(show_info_for=None, show_data_for='all', show_function_for='all', show_band_for='meaningful')

###############
# Plot output #
###############



#Daten bekommen
mnPVC  = my_fits[0].get_parameter_values(rounding=False)
mnfPVC = my_fits[0].get_parameter_errors(rounding=False)
mnMES  = my_fits[1].get_parameter_values(rounding=False)
mnfMES = my_fits[1].get_parameter_errors(rounding=False)
mnCU   = my_fits[2].get_parameter_values(rounding=False)
mnfCU  = my_fits[2].get_parameter_errors(rounding=False)
mnALU  = my_fits[3].get_parameter_values(rounding=True)
mnfALU = my_fits[3].get_parameter_errors(rounding=False)
mnFE   = my_fits[4].get_parameter_values(rounding=False)
mnfFE = my_fits[4].get_parameter_errors(rounding=False)
#E-Modul bestimmen
L=0.454
Lf=0.0005
b=25.0
bf=0.000025
d=0.0060
df=0.000025

'''
def E(m):
    E=(L**3)/(4*b*(d**3)*m)
    return E
'''
mnPVC=np.around(mnPVC[0], decimals=3)
mnMES=np.around(mnMES[0], decimals=3)
mnCU =np.around(mnCU [0], decimals=3)
mnALU=np.around(mnALU[0], decimals=3)
mnFE =np.around(mnFE [0], decimals=3)
'''
def Ef(m,mf):
    Ef=np.sqrt((2.*(L**2)/(4.*b*(d**3)*m))**2*(Lf**2)+(L**3/(4*(b**2)*(d**3)*m))**2*(bf**2)+(L**3/(4*b*(d**3)*(m**2)))**2*(mf**2)+(3*L**3/(4*b*(d**4)*m))**2*(df**2))
    return Ef
'''  
# Fehler auf E berechnen und RUnden  
mnfPVC=np.around(mnfPVC[0],decimals=10)    
mnfMES=np.around(mnfMES[0],decimals=8)   
mnfCU =np.around(mnfCU [0],decimals=10)   
mnfALU=np.around(mnfALU[0],decimals=9)   
mnfFE =np.around(mnfFE [0],decimals=8)   


my_plot.axes.annotate(r'$ m_{PVC} = $' + str(mnPVC), xy=(2.3, 3.8), size=10, ha='right')
my_plot.axes.annotate(r'$ m_{Mes} = $' + str(mnMES), xy=(2.3, 3.6), size=10, ha='right')
my_plot.axes.annotate(r'$ m_{Cu} = $'  + str(mnCU), xy=(2.3, 3.4), size=10, ha='right')
my_plot.axes.annotate(r'$ m_{Al} = $'  + str(mnALU), xy=(2.3, 3.2), size=10, ha='right')
my_plot.axes.annotate(r'$ m_{Fe} = $'  + str(mnFE ), xy=(2.3, 3.0), size=10, ha='right')

my_plot.axes.annotate(r'$\pm $' + str(2.3e-03), xy=(2.32, 3.819), size=10, ha='left')
my_plot.axes.annotate(r'$\pm $' + str(mnfMES), xy=(2.32, 3.619), size=10, ha='left')
my_plot.axes.annotate(r'$\pm $' + str(4.9e-05), xy=(2.32, 3.419), size=10, ha='left')
my_plot.axes.annotate(r'$\pm $' + str(mnfALU), xy=(2.32, 3.219), size=10, ha='left')
my_plot.axes.annotate(r'$\pm $' + str(mnfFE), xy=(2.32, 3.019), size=10, ha='left')

# Save the plots
#my_plot.save('Aufgabe2ueb.pdf')

# Show the plots
my_plot.show()
