# Averaging trajectories on the manifold of covariance matrices

This repository contains the code of the methods proposed in _Averaging trajectories on the manifold of covariance matrices_.

In this repository, we have two notebooks :
- [Trajectory_MDM_BCI_data](Trajectory_MDM_BCI_data.ipynb) : In this notebook, you can find an example of our two proposed algorithms _PT-MDM_ and _DTW_MDM_ tested on a BCI dataset from MOABB [1].
- [Trajectory_MDM_Synthetic_data](Trajectory_MDM_Synthetic_data.ipynb): In this notebook, you can find an example of our two proposed algorithms _PT-MDM_ and _DTW_MDM_ tested on the sythetic datasets descibed in the paper.

In this repository is also provided the core code that we developped for our two methods : _PT-MDM_ and _DTW-MDM_. The codes can be found in the folder `sources`. The come in two versions :
- [`trajectory_mdm`](source/trajectory_mdm.py) : This code is the basic _PT-MDM_ and _DTW-MDM_ that we developped. It is used for the synthetic experiments. One can use this code when the SPD trajectories have already be computed and one juste need to train a classifier from them.
- [`trajectory_fgmdm_for_BCI`](source/trajectory_fgmdm_for_BCI.py) : This code is a slight variant of the previous code : here, we take as input the multivariate time series, then we cut the times series into smaller windows of size `size_window` (a hyperparameter) and compute a covariance matrix for each window. The default covariance estimator used is the sample covariance matrix, but it is a parameter so it can be chosen by the user. Once the covariance matrix trajectories have been computed, we apply the we apply a Fisher Geodesic Discriminant Analysis (FGDA) filter [2] to all of the covariance matrices. (See the paper for more details). This code is used in the experiments on real BCI data.  

### Requirements

Theses are the packages needed in order to run the code.
```
numpy==1.24.2
matplotlib==3.7.1
moabb==0.4.6
pymanopt==2.1.1
scipy==1.10.1
pyriemann==0.4
scikit-learn
tqdm==4.65.0
joblib==1.2.0
```
One can install them using the command :
```
pip install -r requirements.txt
```

### References
[1]: V. Jayaram and A. Barachant, “Moabb: trustworthy algorithm benchmarking for bcis,” _J Neural Eng_, vol. 15, no. 6, pp. 066011, 2018.

[2]: A. Barachant, S. Bonnet, M. Congedo, and C. Jutten, “Riemannian geometry applied to BCI classification,” in _LVA/ICA 2010_. Sept. 2010, vol. 6365, p. 629, Springer.
