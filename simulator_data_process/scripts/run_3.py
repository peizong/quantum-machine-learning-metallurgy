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

# Generate parameter list
pca_list = []
entanglement_list = []
reps_list = []
noise_list = []
classical_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for pca_tmp in [8]:
    for entanglement_tmp in ['circular']:
        for reps_tmp in [2]:
            for noise_tmp in range(16):
                pca_list.append(pca_tmp)
                entanglement_list.append(entanglement_tmp)
                reps_list.append(reps_tmp)
                noise_list.append(noise_tmp)




from qiskit import BasicAer
from warnings import filterwarnings
filterwarnings('ignore')

iter_list_precomputed = []
score_list_precomputed = []
score_train_list_precomputed = []
score_list_classical = []
score_train_list_classical = []
for para_inx in range(len(pca_list)):
    print(pca_list[para_inx], entanglement_list[para_inx], reps_list[para_inx], noise_list[para_inx])
    # Load digits dataset
    data=np.genfromtxt('../alloy-property-avg.csv',delimiter=",",skip_header=1)
    
    # Split as y for phase, X for other properties
    X,y=[],[]
    for i in range(len(data)):
        X.append(data[i][0:-1]), y.append(data[i][-1])
        
    # renormlaised all variables in X
    max_arr=np.max(X,axis=0)
    X=X/max_arr
    
    # Add random matrix (as white noise)
    X_noise=np.random.random((len(data),noise_list[para_inx])) # 1, 5, 10, 15
    Xnew=np.append(X, X_noise, axis=1)
    
    # Split dataset
    sample_train, sample_test, label_train, label_test = train_test_split(Xnew, y, test_size=0.2, random_state=22)

    # Reduce dimensions
    n_dim = pca_list[para_inx]
    pca = PCA(n_components=n_dim).fit(sample_train)
    sample_train = pca.transform(sample_train)
    sample_test = pca.transform(sample_test)

    # Normalise
    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)

    # Scale
    samples = np.append(sample_train, sample_test, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    sample_train = minmax_scale.transform(sample_train)
    sample_test = minmax_scale.transform(sample_test)
    
    # setup of kernel
    encode_map = ZZFeatureMap(feature_dimension=pca_list[para_inx], reps=reps_list[para_inx], entanglement=entanglement_list[para_inx], insert_barriers=True)
    zz_kernel = QuantumKernel(feature_map=encode_map, quantum_instance=BasicAer.get_backend('statevector_simulator'))
    
    # pre-compute
    matrix_train = zz_kernel.evaluate(x_vec=sample_train)
    matrix_test = zz_kernel.evaluate(x_vec=sample_test, y_vec=sample_train)
    
    # Get max of iteration needed
    zzpc_svc = SVC(kernel='precomputed', max_iter=-1)
    zzpc_svc.fit(matrix_train, label_train)
    max_iter = zzpc_svc.n_iter_[0] + 100
    print('Take max_iter as ', max_iter)
    print('Score of validation', zzpc_svc.score(matrix_test, label_test))
    print('Score of training', zzpc_svc.score(matrix_train, label_train))
    
    # loop over iteration from 0 to max_iteration
    iter_list_precomputed_local = []
    score_list_precomputed_local = []
    score_train_list_precomputed_local = []
    for nb_iter in range(max_iter):
        zzpc_svc = SVC(kernel='precomputed', max_iter=nb_iter)
        zzpc_svc.fit(matrix_train, label_train)
        zzpc_score = zzpc_svc.score(matrix_test, label_test)
        zzpc_score_train = zzpc_svc.score(matrix_train, label_train)
        iter_list_precomputed_local.append(nb_iter)
        score_list_precomputed_local.append(zzpc_score)
        score_train_list_precomputed_local.append(zzpc_score_train)
        # print(f'Precomputed kernel classification test score: {zzpc_score}')
        
    # Get score for classical kernals
    score_list_classical_local = []
    score_train_list_classical_local = []
    for kernel_idx, kernel in enumerate(classical_kernels):
        classical_svc = SVC(kernel=kernel, max_iter=-1)
        classical_svc.fit(sample_train, label_train)
        classical_score = classical_svc.score(sample_test, label_test)
        classical_train_score = classical_svc.score(sample_train, label_train)
        score_list_classical_local.append(classical_score)
        score_train_list_classical_local.append(classical_train_score)
    print('Score of validation (classical)', score_list_classical_local)
    print('Score of training (classical)', score_train_list_classical_local)
    iter_list_precomputed.append(iter_list_precomputed_local)
    score_list_precomputed.append(score_list_precomputed_local)
    score_train_list_precomputed.append(score_train_list_precomputed_local)
    score_list_classical.append(score_list_classical_local)
    score_train_list_classical.append(score_train_list_classical_local)

# save results to file
with open('../res/zz_3.npy', 'wb') as f:
    np.save(f, np.array(pca_list))
    np.save(f, np.array(entanglement_list))
    np.save(f, np.array(reps_list))
    np.save(f, np.array(noise_list))
    np.save(f, np.array(iter_list_precomputed))
    np.save(f, np.array(score_list_precomputed))
    np.save(f, np.array(score_train_list_precomputed))
    np.save(f, np.array(score_list_classical))
    np.save(f, np.array(score_train_list_classical))

