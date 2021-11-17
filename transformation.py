'''
Created on Sep 27, 2021
@author: Bart De Wit

'''

csv_flat = 'echt_test2_met video\\flat_surface2.csv' # csv met enkel punten van het grondvlak
directory = 'echt_test2_met video\\frame_2450_2650' # directory met csv's van alle frames
estimated_roll = 0 # geschatte roll-hoek van de LIDAR (kabel is voorkant)
estimated_pitch = 25 # geschatte pitch-hoek van de LIDAR (kabel is voorkant, positief is naar boven)
error = 20 # maximale fout op bovenstaande hoeken
bb = (-30, -5, -20, 40) #bounding box: minx maxx miny maxy

import math
import numpy as np
from numpy import genfromtxt
import os
import matplotlib.pyplot as plt
from matplotlib import animation

data = genfromtxt(csv_flat, delimiter=',') #punten van de straat voor bepalen grondoppervlak
data3d = data[1:, :3] # eerste rij verwijderen, enkel eerste 3 kolommen overhouden

def rotate( punt, ro_matrix ): #een punt roteren op basis van rotatiematrix
    return np.dot(punt,ro_matrix)

def maak_ro_matrix(roll,pitch): #rotatie-matrix maken
    a=0/180*math.pi  # yaw staat op 0
    b=pitch/180*math.pi
    c=roll/180*math.pi #omzetten naar radialen
    ro_matrix=((math.cos(a)*math.cos(b), math.cos(a)*math.sin(b)*math.sin(c)-math.sin(a)*math.cos(c), math.cos(a)*math.sin(b)*math.cos(c)+math.sin(a)*math.sin(c)),
               (math.sin(a)*math.cos(b), math.sin(a)*math.sin(b)*math.sin(c)+math.cos(a)*math.cos(c), math.sin(a)*math.sin(b)*math.cos(c)-math.cos(a)*math.sin(c)),
               (-math.sin(b), math.cos(b)*math.sin(c), math.cos(b)*math.cos(c))) # zie https://en.wikipedia.org/wiki/Rotation_matrix
    return (ro_matrix)

def geef_stafw(arr,roll,pitch): #standaardafwijking van Z-waarde na roteren van puntenwolk
    ro_matrix = maak_ro_matrix(roll, pitch)
    datarot = np.apply_along_axis( rotate, axis=1, arr=arr , ro_matrix = ro_matrix) #voor elke rij (=punt), functie uitvoeren = draaien volgens rotatiematrix
    return (np.std(datarot[:,2], axis=0)) # standaardafwijking van 3e kolom (=z-waarde)

def verfijn(estimated_roll,estimated_pitch,error): 
    # opgegeven: geschatte pitch en roll, en maximale error
    # voor 11 waarden tussen pitch-error en pitch+error standaardafwijking op Z bepalen, idem voor roll. De minimale bijhouden
    min_stafw = 9999  #hoge minimale standaardafwijking
    for i in range(11):
        for j in range(11):
            roll = estimated_roll+(5-j)/10*error*2 #11 waarden voor roll
            pitch = estimated_pitch+(5-i)/10*error*2 #11 waarden voor pitch. Dus in totaal 121 testen
            stafw = geef_stafw(data3d,roll,pitch) # standaardafwijking voor deze combinatie pitch en roll
            if (stafw<min_stafw): #als er een standaardafwijking wordt gevonden die kleiner is
                min_stafw = stafw #dit is de kleinste standaardafwijking...
                new_estimated_roll=roll # ... met deze roll ...
                new_estimated_pitch=pitch # ... en deze pitch
    print (min_stafw)
    return (new_estimated_roll,  new_estimated_pitch)
    
(estimated_roll,estimated_pitch) = verfijn (estimated_roll,estimated_pitch,error) # eerste iteratie
error = error / 10 # maximale error 10 keer kleiner
(estimated_roll,estimated_pitch) = verfijn (estimated_roll,estimated_pitch,error) # tweede iteratie
error = error / 10 # maximale error 10 keer kleiner
(best_roll, best_pitch) = verfijn (estimated_roll,estimated_pitch,error) # derde iteratie
print(best_roll, best_pitch) 

ro_matrix = maak_ro_matrix(best_roll, best_pitch) # rotatiematrix maken van gevonden pitch en roll
datarot = np.apply_along_axis( rotate, axis=1, arr=data3d , ro_matrix = ro_matrix) # de punten van het grondoppervlak nog eens zo draaien
grondz = np.average(datarot[:,2], axis=0) # gemiddelde van de Z-waarde is de hoogte van de grond
print (grondz)

data_vis = []


for filename in os.listdir(directory): # voor alle frames ...
    if filename.endswith(".csv"): # ... met extensie csv
        da = genfromtxt(os.path.join(directory, filename), delimiter=',') # volledige pad-naam
        da =da[da[:, 1] !=0] # waarden met y=0 verwijderen
        da = da[1:, :3] # header verwijderen, eerste 3 kolommen overhouden
        
        da = np.apply_along_axis( rotate, axis=1, arr=da , ro_matrix = ro_matrix) # puntenwolk draaien
    
        print (da.shape)
        
        da =da[da[:, 2] >(grondz+1)] # punten hoger dan 1m boven grond overhouden
        da =da[da[:, 0] >bb[0]] # punten binnen bounding-box overhouden
        da =da[da[:, 0] <bb[1]]
        da =da[da[:, 1] >bb[2]]
        da =da[da[:, 1] <bb[3]]
        
        print (da.shape)
        data_vis.append(da) #puntenwolk opslaan in grote array voor visualisatie
        
fig, ax = plt.subplots(1, 1, figsize = ((bb[1]-bb[0])/10,(bb[3]-bb[2])/10)) # grafiek met grootte ifv bounding-box gegevens

def animate(i):
    ax.cla() # clear the previous image
    ax.scatter(data_vis[i][:,0],data_vis[i][:,1],s=1) # punten tekenen, x en y-waarden
    ax.set_xlim([bb[0], bb[1]]) # fix the x axis
    ax.set_ylim([bb[2], bb[3]]) # fix the y axis

anim = animation.FuncAnimation(fig, animate, frames = len(data_vis) , interval = 100, repeat = True) #animatie maken
plt.show() #animatie tonen
