# quantum-machine-learning-metallurgy

This code is used in our paper titled "Designing complex concentrated alloys with quantum machine learning and text mining".
It consists of tools for (i) data preprocessing, (ii) quantum SVM, and (iii) data postprocessing.


### data preprocessing
The data files and Python script can be found in folder data_preprocessing. Simply run
```shell
python prepare-test-data-alloy-design.py
```
This will generate a file named "test-TM-2864.csv" for the training of quantum SVM models.

### quantum SVM
The quantum SVM model is trained using both quantum simulator BasicAer and IBM quantum computers. This can be achieved by following the steps in QuantumSVM-HEA.ipynb, or submit it as a remote job with qsvm-HEA.py.

### data postprcessing

Please follow the README.md file in the sub-directory simulator_data_process for data post processing.
