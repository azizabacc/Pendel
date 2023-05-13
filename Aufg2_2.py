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


############ def Funktion fur grose Winkel
g=9.81
def T(winkel,r,l):
    Tnull=(2*np.pi*np.sqrt(0.4*r**2+(r+l)**2/(9.81*(l+r))))
    F=(Tnull*(1.+0.25*(np.sin(np.pi*winkel/360.))**2+9./64.*(np.sin(np.pi*winkel/360.))**4))
    return F


###############################
#Einlesen Daten


fname='Aufg2_2.dat'
ncols=2
data_array, info_dict =\
  readColumnData(fname, ncols, delimiter=' ', pr=False)

# print what we got:
phi=data_array[:,0] # 1st column
t=data_array[:,1] # 2nd column


x=np.linspace(5,60,100)
    
fig=plt.figure()
ax1=fig.add_subplot(1,1,1)

plt.plot(phi,t, 'rs',linewidth=1, label='Periodendauer exp.')
plt.plot(phi,t, 'r',linewidth=1)
plt.plot(x,T(x,0.06,2.333), 'b')
plt.plot(phi,T(phi,0.06,2.333), 'bs', label='Periodendauer theo.')
plt.axis([0,65,3.05,3.35])
ax1.set_xlabel('$Auslenkung$ ($^{\circ}$)',size='large')
ax1.set_ylabel('$Periodendauer$ T (s)',size='large')
plt.legend(loc='best')
plt.savefig('Aufg2_2.pdf')


plt.show()
