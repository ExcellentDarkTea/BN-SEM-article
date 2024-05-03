import streamlit as st
import numpy as np
 
import matplotlib.pyplot as plt
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
 
from sklearn.metrics import accuracy_score
 
st.title('Bayesian Network for workload detection')
 
st.write("""
# Run BN
""")
#upload the bn
bn = gum.loadBN("exo1.bif")

#select probability
ie = gum.LazyPropagation(bn)
res_susc = ie.posterior('Workload')

#create sidebar for the sensors

st.sidebar.write("Select the status of the sensors")
# hr_stat = st.sidebar.selectbox("HR", options=["none", "workload", "rest"])
# rr_stat = st.sidebar.selectbox("RR", options=["none", "workload", "rest"])
# temp_stat = st.sidebar.selectbox("Temp", options=["none", "workload", "rest"])
# gsr_stat = st.sidebar.selectbox("GSR", options=["none", "workload", "rest"])
# statuses = [hr_stat, rr_stat, temp_stat, gsr_stat]

param_names = ["HR", "RR", "Temp", "GSR"]
option = ["none", "workload", "rest"]
statuses = {}
params = dict()
#-----------------------------------
# create sidebar for the sensors
for param in param_names:
    statuses[param] = st.sidebar.selectbox(param, options=option)

def set_param(param_names, statuses, params):
    # params = dict()
    for param, stat in zip(param_names, statuses):
        if stat == "workload":
            params[param] = 1
        elif stat == "rest":
            params[param] = 0
    return params

#-----------------------------------
# create sidebar for the demographic
st.sidebar.write("Select the status of demographic")
option = ["none", "male", "female"]
sex = st.sidebar.selectbox("Sex", options=option)
if sex == "male":
    params["Sex"] = 0
elif sex == "female":
    params["Sex"] = 1 

option = ["none", "18-24", "25-34", "35-64"]
age = st.sidebar.selectbox("Age", options=option)
if age == "18-24":
    params["Age"] = 0
elif age == "25-34":    
    params["Age"] = 1   
elif age == "35-64":
    params["Age"] = 2

#-----------------------------------
# create sidebar for the Psychology traits
st.sidebar.write("Select the status of the pcychological traits")
param_names_pcy = ["diligence", "prudence","flexibility","gentleness","patience","fairness","greed_avoidance","modesty","sincerity","liveliness","sociability","social_boldness","openess","dependence"]
option = ["none", "1", "2", "3", "4", "5"]
statuses_pcy = {}

for param in param_names_pcy:
    statuses_pcy[param] = st.sidebar.selectbox(param, options=option)

def set_param_pcy(param_names_pcy, statuses_pcy, params):
    # params = dict()
    for param, stat in zip(param_names_pcy, statuses_pcy):
        if stat == "1":
            params[param] = 0
        elif stat == "2":
            params[param] = 1
        elif stat == "3":
            params[param] = 2
        elif stat == "4":
            params[param] = 3
        elif stat == "5":
            params[param] = 4
    return params

params = set_param(param_names, statuses.values(), params)
params_pcy = set_param_pcy(param_names_pcy, statuses_pcy.values(), params)


evs = params
gnb.showInference(bn,evs=evs,size='45')
ie = gum.LazyPropagation(bn)
ie.setEvidence(evs)
 
res_susc = ie.posterior('Workload')
st.write(f'P(Workload): {round(res_susc[1] * 100, 2)}')

st.write(params)
# node_names = [bn.variable(i).name() for i in bn.nodes()]
# st.write(node_names)


