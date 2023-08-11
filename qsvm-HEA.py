
# General Imports
import numpy as np

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
################################################
# Load digits dataset
data=np.genfromtxt('alloy-property-avg.csv',delimiter=",",skip_header=1)
#design_data=np.genfromtxt('test-TM.csv',delimiter=",",skip_header=1)
#design_data=design_data[:,2:]

#digits = datasets.load_digits(n_class=2)
print(len(data))
print(data[0])
print(data[0][0:-1])
X,y=[],[]
for i in range(len(data)):
    X.append(data[i][0:-1]),y.append(data[i][-1])
print("X,y: ",X[0],y[0])
max_arr=np.max(X,axis=0)
print("max_arr: ",max_arr,X/max_arr)
X=X/max_arr
print("renormalized X: ",X)
X5=np.random.random((len(data),15))
print("X5: ",X5)
Xnew=np.append(X, X5, axis=1)
print("Xnew: ",Xnew)
# Plot example '0' and '1'
#fig, axs = plt.subplots(1, 2, figsize=(6,3))
#axs[0].set_axis_off()
#axs[0].imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
#axs[1].set_axis_off()
#axs[1].imshow(digits.images[100], cmap=plt.cm.gray_r, interpolation='nearest')
#plt.show()

################################################
# Split dataset
#sample_train, sample_test, label_train, label_test = train_test_split(
#     digits.data, digits.target, test_size=0.2, random_state=22)
sample_train, sample_test, label_train, label_test = train_test_split(
    X,y, test_size=0.2, random_state=22)
print(sample_train[0])

# Reduce dimensions
n_dim = 4 #6 #5 #4
pca = PCA(n_components=n_dim).fit(sample_train)
sample_train = pca.transform(sample_train)
sample_test = pca.transform(sample_test)
print(sample_train[0])
#design_pca=PCA(n_components=n_dim).fit(design_data)
#design_data=design_pca.transform(design_data)

# Normalise
std_scale = StandardScaler().fit(sample_train)
sample_train = std_scale.transform(sample_train)
sample_test = std_scale.transform(sample_test)
print(sample_train[0])
#design_std_scale = StandardScaler().fit(design_data)
#design_data = design_std_scale.transform(design_data)
#print("normalized design_data[0]", design_data[0])

# Scale
samples = np.append(sample_train, sample_test, axis=0)
minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
sample_train = minmax_scale.transform(sample_train)
sample_test = minmax_scale.transform(sample_test)
print("sample_train[0]: ",sample_train[0])
#design_minmax_scale = MinMaxScaler((-1, 1)).fit(design_data)
#design_data = design_minmax_scale.transform(design_data)
#print("scaled design_data[0]: ",design_data[0])

# Select
train_size = 400 #100 #*2*2
sample_train = sample_train[:train_size]
label_train = label_train[:train_size]

test_size = 100 #20 #*2*2
sample_test = sample_test[:test_size]
label_test = label_test[:test_size]
#design_data=design_data[:test_size]
################################################
#zz_map = ZZFeatureMap(feature_dimension=4, reps=2, entanglement='circular', insert_barriers=True)

################################################

#----------------ibm quantum computer---------
# The second part is copied from this website:
#https://qiskit.org/ecosystem/ibm-provider/tutorials/1_the_ibm_quantum_account.html

from qiskit_ibm_provider import IBMProvider
TOKEN="0823eb968ee0224cbc7f65361596ac0321f48ac304942867d7ede1f39e4bbef2230de59a0dbf81af65c93531e449c6a2d87f543b594a17ca96a4b4f5beef1459" # Pei
#TOKEN="c300ca2fc313714769bf564f48ce59f9c1c8e332ee287faee555108ed65a47effaa64dc3dcd2bbe45899436ae2162ad763d6a48502b276e8e32f8758cbf590cf"  # Mine
IBMProvider.save_account(TOKEN, overwrite=True) #uncomment this line for the first-time calculation
provider = IBMProvider()
provider.backends()

from qiskit.compiler import transpile, assemble
from qiskit_ibm_provider import least_busy

small_devices = provider.backends(min_num_qubits=4, simulator=False, operational=True)
backend = least_busy(small_devices)
print(backend)

import time
time1=time.time()
print("time begin: ",time1)
################################################

backend = least_busy(small_devices)
zz_map = ZZFeatureMap(feature_dimension=4, reps=2, entanglement='circular', insert_barriers=True)
zz_kernel = QuantumKernel(feature_map=zz_map, quantum_instance=backend)

zz_circuit = zz_kernel.construct_circuit(sample_train[0], sample_train[1])
#zz_circuit = zz_kernel.construct_circuit_with_feature_map(sample_train[0], sample_train[1])
zz_circuit.decompose().decompose().draw(output='mpl')

mapped_circuit = transpile(zz_circuit, backend=backend)
job = backend.run(mapped_circuit, shots=1024*4)
################################################

print(job.status())
result = job.result()
counts = result.get_counts()
print(counts)
################################################
matrix_train = zz_kernel.evaluate(x_vec=sample_train)
matrix_test = zz_kernel.evaluate(x_vec=sample_test, y_vec=sample_train)
#matrix_design = zz_kernel.evaluate(x_vec=design_data, y_vec=sample_train)
################################################
zzpc_svc = SVC(kernel='precomputed')
zzpc_svc.fit(matrix_train, label_train)
zzpc_score = zzpc_svc.score(matrix_test, label_test)

print(f'Precomputed kernel classification test score: {zzpc_score}')
time2=time.time()
print("time used: ",time2-time1)
################################################

zzcb_svc = SVC(kernel=zz_kernel.evaluate)
#zzcb_svc = SVC(kernel=mapped_circuit.evaluate)
zzcb_svc.fit(sample_train, label_train)
zzcb_score = zzcb_svc.score(sample_test, label_test)

print(f'Callable kernel classification test score: {zzcb_score}')
time3=time.time()
print("time used: ",time3-time1)
design=False
if design==True:
  pred=zzcb_svc.predict(design_data)
  #pred=zzcb_svc.predict(sample_test)

  df=pd.DataFrame(pred)
  df.to_csv("prediction-qsvm-hardware.csv")

  print("test_label for comparison:", label_test)

