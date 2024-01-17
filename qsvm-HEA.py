
# General Imports
import numpy as np
import pandas as pd

# Visualisation Imports
import matplotlib.pyplot as plt

# Scikit Imports
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Qiskit Imports
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap
from qiskit.circuit.library import TwoLocal, NLocal, RealAmplitudes, EfficientSU2
from qiskit.circuit.library import HGate, RXGate, RYGate, RZGate, CXGate, CRXGate, CRZGate
from qiskit_machine_learning.kernels import QuantumKernel
from sklearn.utils import shuffle
################################################
# Load digits dataset
#data=np.genfromtxt('alloy-property-avg.csv',delimiter=",",skip_header=1)
data=np.genfromtxt('training-data.csv',delimiter=",",skip_header=1)
data=shuffle(data, random_state=10) 
data=np.delete(data,15,1) #CrustAbundance
print("training data: ",data[0])
design_data=np.genfromtxt('test-TM-25476.csv',delimiter=",",skip_header=1)
design_data=design_data[:,2:]
#digits = datasets.load_digits(n_class=2)
print("design_data[0]: ",design_data[0])
print("design_data[1][1:-1]: ",design_data[1][1:-1])

X,y=[],[]
for i in range(len(data)):
    X.append(data[i][0:-1]),y.append(data[i][-1])
print("X,y: ",X[0],y[0])
max_arr=np.max(X,axis=0)
X=X/max_arr
print("max_arr: ",max_arr,X)

#normalize the design data
max_arr=np.max(design_data,axis=0)
design_data = design_data/max_arr
print("renormalized design_data: ",design_data)

# test the effect of noise by adding several columns of white noise
# X5=np.random.random((len(data),15))
# print("X5: ",X5)
# Xnew=np.append(X, X5, axis=1)
# print("Xnew: ",Xnew)


################################################
# Split dataset
#sample_train, sample_test, label_train, label_test = train_test_split(
#     digits.data, digits.target, test_size=0.2, random_state=22)
sample_train, sample_test, label_train, label_test = train_test_split(
    X,y, test_size=0.2, random_state=22)
#print(sample_train[0])

# Reduce dimensions
n_dim = 8 #6 #4 #8 #4 #6 #5 #4
pca = PCA(n_components=n_dim).fit(X) #(sample_train)
sample_train = pca.transform(sample_train)
sample_test = pca.transform(sample_test)
#print(sample_train[0])
#design_pca=PCA(n_components=n_dim).fit(design_data)
#design_data=design_pca.transform(design_data)
design_data=pca.transform(design_data)

# # Normalise
# std_scale = StandardScaler().fit(sample_train)
# sample_train = std_scale.transform(sample_train)
# sample_test = std_scale.transform(sample_test)
# design_std_scale = StandardScaler().fit(design_data)
# design_data = design_std_scale.transform(design_data)
# print("normalized design_data[0]", design_data[0])
# Scale
samples = np.append(sample_train, sample_test, axis=0)
minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
sample_train = minmax_scale.transform(sample_train)
sample_test = minmax_scale.transform(sample_test)
# design_minmax_scale = MinMaxScaler((-1, 1)).fit(design_data)
# design_data = design_minmax_scale.transform(design_data)
design_data = minmax_scale.transform(design_data)
print("scaled design_data[0]: ",design_data[0])

## Select
#train_size = 100
#sample_train = sample_train[:train_size]
#label_train = label_train[:train_size]

#test_size = 20
#sample_test = sample_test[:test_size]
#label_test = label_test[:test_size]

################################################
from qiskit import BasicAer
zz_map = ZZFeatureMap(feature_dimension=n_dim, reps=2, entanglement='linear', insert_barriers=True)
#zz_map = ZZFeatureMap(feature_dimension=n_dim, reps=1, entanglement='circular', insert_barriers=True)
#zz_map = ZFeatureMap(feature_dimension=6, reps=3)
#zz_map = PauliFeatureMap(feature_dimension=6, reps=3, paulis = ['X', 'Y', 'ZZ'])
#zz_map = TwoLocal(num_qubits=6, reps=2, rotation_blocks=['ry','rz'],
#               entanglement_blocks='cx', entanglement='circular', insert_barriers=True)
#zz_kernel = QuantumKernel(feature_map=zz_map, quantum_instance=Aer.get_backend('statevector_simulator'))
zz_kernel = QuantumKernel(feature_map=zz_map, quantum_instance=BasicAer.get_backend('statevector_simulator'))
print(sample_train[0])
print(sample_train[1])
zz_circuit = zz_kernel.construct_circuit(sample_train[0], sample_train[1])
#zz_circuit = zz_kernel.construct_circuit_with_feature_map(sample_train[0], sample_train[1])
zz_circuit.decompose().decompose().draw(output='mpl')

#----------------simulator--------------------
#backend = Aer.get_backend('qasm_simulator')
backend = BasicAer.get_backend('qasm_simulator')

job = execute(zz_circuit, backend, shots=8192,
              seed_simulator=1024, seed_transpiler=1024)
counts = job.result().get_counts(zz_circuit)

################################################

################################################

################################################

matrix_train = zz_kernel.evaluate(x_vec=sample_train)
matrix_test = zz_kernel.evaluate(x_vec=sample_test, y_vec=sample_train)
#matrix_design = zz_kernel.evaluate(x_vec=design_data, y_vec=sample_train) #not used???
################################################
zzpc_svc = SVC(kernel='precomputed')
zzpc_svc.fit(matrix_train, label_train)
zzpc_score = zzpc_svc.score(matrix_test, label_test)

print(f'Precomputed kernel classification test score: {zzpc_score}')
################################################

zzcb_svc = SVC(kernel=zz_kernel.evaluate)
#zzcb_svc = SVC(kernel=mapped_circuit.evaluate)
zzcb_svc.fit(sample_train, label_train)
zzcb_score = zzcb_svc.score(sample_test, label_test)
# val=zzpc_svc.predict(np.transpose(sample_test))
# val=np.concatenate((val,label_test))
# df=pd.DataFrame(val)
# df.to_csv("val_pred.csv")
print(f'Callable kernel classification test score: {zzcb_score}')

val=zzcb_svc.predict(sample_test)
#val=np.concatenate((val,label_test),axis=1)
val=np.stack((val, label_test), axis=1)
df=pd.DataFrame(val)
df.to_csv("val_pred.csv")

print("sample_test: ", sample_test[0])
print("design_data: ", design_data[0])

pred=zzcb_svc.predict(design_data)
df=pd.DataFrame(pred)
df.to_csv("prediction-qsvm-simulator.csv")
