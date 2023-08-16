#!/bin/bash

import numpy as np
import pandas as pd
import re

def extract_element_conc(strings):
  element,concentration=[],[]
  r=re.compile('([A-Za-z]+)([-?\d+\.\d+|\d+]+)')
  while (strings !=''):
    m=r.match(strings)
    element.append(m.group(1))
    concentration.append(float(m.group(2)))
    short_st= m.group(1)+m.group(2)
    strings=strings.replace(short_st,'')
  concentration /= np.sum(concentration)
  return [element,concentration]

def read_data(filename):
  dat=pd.read_csv(filename)
  #print(dat)
  #name of alloys
  #alloys=dat["sys1"]
  return dat

def prepare_test(data,dictionary):
  new_data=[]
  delta=[]
  count_1=0
  col=["AtomicWeight","BoilingPoint","BulkModulus","el_neg","CovalentRadius",\
           "CrustAbundance","Density","FusionHeat","MeltingPoint","MolarVolume",\
           "NeutronCrossSection","SpecificHeat","ThermalConductivity","Valence",\
           "VaporizationHeat"]
  col_names=["Abbreviation"]+col
  for i in range(0,len(data)):
     ll=data[i]
     alloy=extract_element_conc(ll)
     conc=alloy[1]
     features=[]
     for col_i in col: features.append(np.zeros(len(alloy[0])))
     for j in range(0,len(alloy[0])):
        k=0
        while (alloy[0][j] !=dictionary["Abbreviation"][k] and k<len(dictionary[col[2]])):
          k +=1
        for col_i in range(0,len(col)): features[col_i][j]=dictionary[col[col_i]][k]
        #Tm[j] +=273.16 #use Kelvin for tempearture
     features_avg=[]
     features_avg.append(data[i])
     for col_i in range(0,len(col)): features_avg.append(np.dot(features[col_i],conc)) 
     new_data.append(features_avg)
  df=pd.DataFrame(new_data)
  df.columns=col_names
  return df #new_data #[new_criterion,count_1,delta]
def write_to_file(dat,filename):
  #df=pd.DataFrame(dat)
  dat.to_csv(filename)

if __name__=="__main__":
  num_alloys=2864 
  TM_results="selected-HEAs-equiatomic-lightweight-"+str(num_alloys)+".csv"
  alloys_TM=read_data(TM_results)['sys1']
  print("alloys_TM: ",alloys_TM[0])
  alloy="Fe1Co1" #Ni1Cr1Mn1"
  element,concentration=extract_element_conc(alloy)
  print(element,concentration)
  filename="elementalProperties.csv"
  dictionary=read_data(filename)
  #print("dictionary: ",dictionary)
  data=alloys_TM #[0:10] #[alloy]
  prepared_test=prepare_test(data,dictionary)
  write_to_file(prepared_test,"test-TM-"+str(num_alloys)+".csv")
