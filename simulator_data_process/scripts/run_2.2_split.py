# General Imports
import numpy as np
import sys

# Scikit Imports
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Qiskit Imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit import BasicAer
from warnings import filterwarnings

# Get noise_index as pass in parameter
rep_test = int(str(sys.argv[1]))
print('rep_test', rep_test)

# Generate parameter list
pca_list = []
entanglement_list = []
reps_list = []
classical_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
classical_kernels_text = ['Linear', 'Poly', 'RBF', 'Sigmoid']
for pca_tmp in [8]:
    for entanglement_tmp in ['circular']:
        for reps_tmp in [rep_test]:
            pca_list.append(pca_tmp)
            entanglement_list.append(entanglement_tmp)
            reps_list.append(reps_tmp)

filterwarnings('ignore')

nb_rand_sampling = 300
para_inx = 0
print(pca_list[para_inx], entanglement_list[para_inx], reps_list[para_inx])
# Randomly sample dataset
iter_list_precomputed_local = []
score_list_precomputed_local = []
score_train_list_precomputed_local = []
score_list_classical_local = []
score_train_list_classical_local = []
for attemp_i in range(nb_rand_sampling):
    # Load digits dataset
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
    encode_map = ZZFeatureMap(feature_dimension=pca_list[para_inx], reps=reps_list[para_inx], entanglement=entanglement_list[para_inx],
                              insert_barriers=True)
    zz_kernel = QuantumKernel(feature_map=encode_map, quantum_instance=BasicAer.get_backend('statevector_simulator'))

    # pre-compute
    matrix_train = zz_kernel.evaluate(x_vec=sample_train)
    matrix_test = zz_kernel.evaluate(x_vec=sample_test, y_vec=sample_train)

    # Get max of iteration needed
    zzpc_svc = SVC(kernel='precomputed', max_iter=-1)
    zzpc_svc.fit(matrix_train, label_train)
    max_iter = zzpc_svc.n_iter_[0] + 100
    print('Attempt ', attemp_i)
    print('Take max_iter as ', max_iter)
    print('Score of validation', zzpc_svc.score(matrix_test, label_test))
    print('Score of training', zzpc_svc.score(matrix_train, label_train))

    # loop over iteration from 0 to max_iteration
    iter_list_precomputed_local_local = []
    score_list_precomputed_local_local = []
    score_train_list_precomputed_local_local = []
    for nb_iter in range(max_iter):
        zzpc_svc = SVC(kernel='precomputed', max_iter=nb_iter)
        zzpc_svc.fit(matrix_train, label_train)
        zzpc_score = zzpc_svc.score(matrix_test, label_test)
        zzpc_score_train = zzpc_svc.score(matrix_train, label_train)
        iter_list_precomputed_local_local.append(nb_iter)
        score_list_precomputed_local_local.append(zzpc_score)
        score_train_list_precomputed_local_local.append(zzpc_score_train)
        # print(f'Precomputed kernel classification test score: {zzpc_score}')

    # Get score for classical kernels
    score_list_classical_local_local = []
    score_train_list_classical_local_local = []
    for kernel_idx, kernel in enumerate(classical_kernels):
        classical_svc = SVC(kernel=kernel, max_iter=-1)
        classical_svc.fit(sample_train, label_train)
        classical_score = classical_svc.score(sample_test, label_test)
        classical_train_score = classical_svc.score(sample_train, label_train)
        score_list_classical_local_local.append(classical_score)
        score_train_list_classical_local_local.append(classical_train_score)
    print('Score of validation (classical)', score_list_classical_local_local)
    print('Score of training (classical)', score_train_list_classical_local_local)
    iter_list_precomputed_local.append(iter_list_precomputed_local_local)
    score_list_precomputed_local.append(score_list_precomputed_local_local)
    score_train_list_precomputed_local.append(score_train_list_precomputed_local_local)
    score_list_classical_local.append(score_list_classical_local_local)
    score_train_list_classical_local.append(score_train_list_classical_local_local)

    # save results to file
    filename = '../res/zz_2.2_' + str(rep_test) + '.npy'
    with open(filename, 'wb') as f:
        np.save(f, np.array(pca_list))
        np.save(f, np.array(entanglement_list))
        np.save(f, np.array(reps_list))
        np.save(f, np.array(iter_list_precomputed_local))
        np.save(f, np.array(score_list_precomputed_local))
        np.save(f, np.array(score_train_list_precomputed_local))
        np.save(f, np.array(score_list_classical_local))
        np.save(f, np.array(score_train_list_classical_local))
