# General Imports
import numpy as np
import sys

# Scikit Imports
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Qiskit Imports
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit import BasicAer
from warnings import filterwarnings

# Get data_size as pass in parameter
case_test = int(str(sys.argv[1]))
print('case_test', case_test)

# Generate parameter list
nb_rand_sampling = 300
pca_list = [8, 8, 8, 6, 6, 6]
entanglement_list = ['linear', 'circular', 'none', 'linear', 'circular', 'none']
reps_list = [1, 1, 1, 2, 2, 2]
noise_list = [0, 0, 0, 0, 0, 0]
classical_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
classical_kernels_text = ['Linear', 'Poly', 'RBF', 'Sigmoid']

filterwarnings('ignore')

print(pca_list[case_test], entanglement_list[case_test], reps_list[case_test], noise_list[case_test])
# Randomly sample dataset
score_list_precomputed_local = []
score_train_list_precomputed_local = []
score_list_classical_local = []
score_train_list_classical_local = []
for attemp_i in range(nb_rand_sampling):
    print('Sample', attemp_i)
    # Read data from csv file
    data = np.genfromtxt('../alloy-property-avg.csv', delimiter=",", skip_header=1)

    # Split as y for phase, X for other properties
    X, y = [], []
    for i in range(len(data)):
        X.append(data[i][0:-1]), y.append(data[i][-1])

    # renormlaised all variables in X
    max_arr = np.max(X, axis=0)
    X = X / max_arr

    # Split dataset
    sample_train, sample_test, label_train, label_test = train_test_split(X, y, test_size=0.2, random_state=22 + attemp_i)

    # Reduce dimensions
    n_dim = pca_list[case_test]
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
    if entanglement_list[case_test] == 'none':
        encode_map = ZZFeatureMap(feature_dimension=pca_list[case_test], reps=reps_list[case_test], entanglement=None, insert_barriers=True)
    else:
        encode_map = ZZFeatureMap(feature_dimension=pca_list[case_test], reps=reps_list[case_test], entanglement=entanglement_list[case_test],
                                  insert_barriers=True)
    zz_kernel = QuantumKernel(feature_map=encode_map, quantum_instance=BasicAer.get_backend('statevector_simulator'))

    # pre-compute
    matrix_train = zz_kernel.evaluate(x_vec=sample_train)
    matrix_test = zz_kernel.evaluate(x_vec=sample_test, y_vec=sample_train)

    # Get max of iteration needed
    zzpc_svc = SVC(kernel='precomputed', max_iter=-1)
    zzpc_svc.fit(matrix_train, label_train)
    max_iter = zzpc_svc.n_iter_[0] + 100
    train_score = zzpc_svc.score(matrix_train, label_train)
    validation_score = zzpc_svc.score(matrix_test, label_test)
    score_list_precomputed_local.append(validation_score)
    score_train_list_precomputed_local.append(train_score)
    print('Attempt ', attemp_i)
    print('Take max_iter as ', max_iter)
    print('Score of validation', validation_score)
    print('Score of training', train_score)

    # save results to file
    filename = '../res/zz_7_' + str(case_test) + '.npy'
    with open(filename, 'wb') as f:
        np.save(f, np.array(pca_list))
        np.save(f, np.array(entanglement_list))
        np.save(f, np.array(reps_list))
        np.save(f, np.array(noise_list))
        np.save(f, np.array(score_list_precomputed_local))
        np.save(f, np.array(score_train_list_precomputed_local))
