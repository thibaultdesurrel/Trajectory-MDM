import numpy as np
import matplotlib.pyplot as plt

import pymanopt.optimizers
import pymanopt.manifolds
from scipy.linalg import expm

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.extmath import softmax

from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.base import logm
from pyriemann.utils.mean import mean_riemann
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import FGDA

from tqdm import tqdm

from joblib import Parallel, delayed

import scipy.stats as stat

from utils import *


class DTW_MDM_Temp(BaseEstimator, ClassifierMixin):
    """Classification using the Riemannian DTW to compute the mean trajectory

    Classification of covariances trajectories. First, the trajectories are created by cutting the original time series in smaller windows of size size_window.
    One covariance matrix is computed per window to form a trajectory. Then for each class, the mean trajectory is computed using a iterative two step algorithm :
    First, a matching between each training trajectory and the current mean one is computed using Riemannian DTW. Then, a weighted Fréchet mean is
    computed to update the mean trajectory. Finally, the classification uses a minimum Riemannian DTW distance to mean algorithm to affect the new trajectory
    to the closest averaged trajectory.

    Parameters
    ------------
    manifold : pymanopt.manifold
        The manifold on which the data lies on
    nb_it : int
        Number of iteration for the computation of the mean trajectories
    eps : float
        Tolerance between two consecutive mean trajectories
    size_mean_traj : int
        The number of points on the mean trajectory
    size_window : int
        The size of the windows used to create the n_trajectories
    overlap : int
        The overlap of each window used to create the n_trajectories
    optimizer : pymanopt.optimizer, default = pymanopt.optimizers.SteepestDescent
        The pymanopt optimizer used to solve the minimization problem
    cov_estimator : str, default = "scm"
        The covariance estimator used to estimate the covariance of each window on the trajectories. See pyriemann's doc
    """

    def __init__(
        self,
        manifold,
        nb_it,
        eps,
        size_mean_traj,
        size_window,
        overlap = 0,
        optimizer=pymanopt.optimizers.SteepestDescent(verbosity=0),
        cov_estimator="scm",
    ):
        """Init."""
        self.manifold = manifold
        self.nb_it = nb_it
        self.eps = eps
        self.size_mean_traj = size_mean_traj
        self.size_window = size_window
        self.optimizer = optimizer
        self.cov_estimator = cov_estimator
        self.overlap = overlap
        self.loss = None
        self.fgda = FGDA()

    def get_temporal_trajectories(self, X, width_bins, overlap):
        """ Creates the subwindows of X in order to compute the covariance trajectories.

        Parameters
        -------------
        X : array_like of size (n_channels, N)
            The original time series
        width_bins : int
            The size of the windows that will be extracted from X
        overlap : int
            The overlap that will be used to create the windows

        Returns
        -------------
        traj : array_like of size (n_windows, n_channels, width_bins)
            The array containing all the windows extracted

        """
        traj = []
        t = 0
        N = X.shape[1]
        while t < N:
            traj.append(X[:, t : (t + width_bins)])
            t += width_bins - overlap
        # In order to have all the windows of same length, we do not return the last window that is often smaller
        return np.array(traj[:-1])

    def preprocess_data(self, X, y=None):
        """ Preprocess the data in order to create the trajectories of covariance matrices.

        Parameters
        -------------
        X : array_like of size (n_series, n_channels, length_series)
            The original time series
        y : array_like of size (n_series,)
            The labels of each trajectory of X

        Returns
        -------------
        all_traj_filtred : array_like of size (n_series, length_traj, n_channels, n_channels)
            The array containing all the covariance matrix trajectories filtred using a FGDA filter.
        """

        # We start by extracting all the subwindows of the main time series X to create the trajectories
        traj_temp = []
        for i in range(X.shape[0]):
            traj_temp.append(self.get_temporal_trajectories(X[i], self.size_window, self.overlap))
        traj_temp = np.array(traj_temp)

        # Once the subwindows are extracted, we can compute one covariance matrix per sub window
        # therefore creating the covariance matrix trajectories.
        (nb_traj, length_traj, nb_sensors, length_EGG) = traj_temp.shape
        # print(length_traj)
        all_traj = np.zeros((nb_traj, length_traj, nb_sensors, nb_sensors))
        for j in range(length_traj):
            all_traj[:, j] = Covariances(estimator=self.cov_estimator).transform(
                traj_temp[:, j, :, :]
            )

        # Then we filter the trajectories
        if y is None:
            # If y is None, then the estimator (therefore the FGDA filter) has already been fitted,
            # so we just need to transform the data
            all_traj_filtred = self.fgda.transform(
                np.reshape(
                    all_traj,
                    (
                        all_traj.shape[0] * all_traj.shape[1],
                        all_traj.shape[2],
                        all_traj.shape[2],
                    ),
                )
            )
            all_traj_filtred = np.reshape(
                all_traj_filtred,
                (
                    all_traj.shape[0],
                    all_traj.shape[1],
                    all_traj.shape[2],
                    all_traj.shape[2],
                ),
            )
        else:
            # Otherwise, we need to fit the fgda filter using all the covariance matrices we just computed
            big_y = np.array(
                [y[i] for i in range(len(y)) for j in range(all_traj.shape[1])]
            )
            all_traj_filtred = self.fgda.fit_transform(
                np.reshape(
                    all_traj,
                    (
                        all_traj.shape[0] * all_traj.shape[1],
                        all_traj.shape[2],
                        all_traj.shape[2],
                    ),
                ),
                big_y,
            )
            all_traj_filtred = np.reshape(
                all_traj_filtred,
                (
                    all_traj.shape[0],
                    all_traj.shape[1],
                    all_traj.shape[2],
                    all_traj.shape[2],
                ),
            )

        return all_traj_filtred

    def random_trajectories(self, T, all_M):
        """Generate a random trajectory of SPD matrices of size T.
        The trajectory will we created along the geodesics of the successive matrices of all_M.

        Parameters
        -------------
        T : int
            Number of points on the trajectory
        all_M : array_like of size (n_M, n_channels, n_channels)
            SPD matrices that will be used the generate the trajectory

        Returns
        -------------
        traj : array_like of size (T, n_channels, n_channels)
            The random trajectory generated
        """
        n_M = len(all_M)
        n = all_M[0].shape[0]
        # We start by defining the times of each points
        all_t = stat.uniform.rvs(size=T)
        all_t = np.sort(all_t)
        traj = []
        # We need to know in which subinterval lies each time t in all_t
        for t in all_t:
            i = 0
            # While t is not in the interval [i/(n_M - 1), (i+1)/(n_M - 1)], we increment i
            while t > (i + 1) / (n_M - 1):
                i += 1
            # Then we can compute a SPD matrix on the geodesic between all_M[i] and all_M[i + 1]
            v = self.manifold.log(all_M[i], all_M[i + 1])
            # We add a small random noise
            epsilon = stat.norm.rvs(scale=0.5, size=n)
            # We rescale t in [0,1]
            t_tilde = (t - i / (n_M - 1)) * ((n_M - 1))
            # Add the new point to the trajectory
            traj.append(
                exp(all_M[i], t_tilde * v)
                + epsilon.reshape(n, 1) @ epsilon.reshape(1, n)
            )
        traj = np.array(traj)
        return traj


    def create_cost(self, all_traj, A, f):
        """Create the cost function as well as its Riemannian gradient associated to our problem

        Parameters
        -------------
        all_traj : array_like of size (n_trajectories, n_matrices, n_channels, n_channels)
            All the trajectories we want to average
        A : array_like of size (n_trajectories, n_channels, n_channels)
            The weights used to compute the weighted mean
        f : int
            The index of the element we want to mean in all_traj
        Returns
        -------------
        cost : function
            The cost function associated to our problem
        riemannian_gradient : function
            The Riemannian gradient of the cost function associated to our problem
        """
        (N, F) = all_traj.shape[:2]

        @pymanopt.function.numpy(self.manifold)
        def cost(X):
            # The cost is \sum_{i = 1}^N \sum_{f' = 1}^F \alpha_{i}^{f,f'} \delta^2(X,X_i^{f'})
            return np.sum(
                [
                    A[i, f, f_prime]
                    * np.power(distance_riemann(X, all_traj[i, f_prime]), 2)
                    for i in range(N)
                    for f_prime in range(F)
                ]
            )

        @pymanopt.function.numpy(self.manifold)
        def riemannian_gradient(X):
            # The gradient is -2\sum_i \sum_{f'} \alpha_{i}^{f,f'} Log_X(X_i^f')
            # We try to compute the Log smartly by computing the inverse of X only once
            c = np.linalg.cholesky(X)
            c_inv = np.linalg.inv(c)
            return -2 * np.sum(
                [
                    A[i, f, f_prime]
                    * (c @ logm(c_inv @ all_traj[i, f_prime] @ c_inv.T) @ c.T)
                    for i in range(N)
                    for f_prime in range(F)
                ],
                axis=0,
            )

        return cost, riemannian_gradient

    def get_coeff_from_DTW(self, matching, F):
        """Get the coefficients alpha_i^{f,f'} from the DTW matching

        The coefficient alpha_i^{f,f'} is the influence of the f'th point of the ith trajectory on the fth point of the mean trajectory.
        Here A[f,f_prime] is alpha_i^{f,f'}.
        Parameters
        -------------
        matching : array_like of size (n_matchings, 2)
            The array of all matching from the DTW algorithm
        F : int
            The number of points in the trajectory (not the mean one)

        Returns
        -------------
        A : array_like of size (size_mean_traj, F)
            The coefficients $\alpha_i^{f,f'}$
        """

        A = np.zeros((self.size_mean_traj, F))
        for f in range(self.size_mean_traj):
            for f_prime in range(F):
                if (f_prime, f) in matching:
                    A[f, f_prime] = 1 / len(np.where(np.array(matching)[:, 1] == f)[0])
        return A

    def compute_mean_traj_DTW(self, all_traj):
        """Compute the mean trajectory of the trajectories in all_traj

        Parameters
        -------------
        all_traj : array_like of size (n_trajectories, n_matrices, n_channels, n_channels)

        Returns
        -------------
        traj_bar : array_like of size (n_matrices, n_channels, n_channels)
            The mean trajectory
        loss : array_like of size (self.nb_it)
            The loss at each step of the iterative algorithm
        stop_crit : str
            The reason the algorithm stopped : it can be because it has reach its number of iteration self.nb_it ("it")
            or if two the norm of the difference of two successive trajectories is smaller than self.eps ("conv")

        """
        (N, F, C) = all_traj.shape[:3]
        all_traj_bar = np.zeros((self.nb_it + 1, self.size_mean_traj, C, C))
        loss = []

        # We start by initializing the mean trajectory randomly
        # For this, we select a random trajectory in the training set and we sample a random trajectory along the geodesic
        # joining the first and last point of this trajectory
        idx = np.random.randint(N)
        traj_bar = self.random_trajectories(
            self.size_mean_traj, [all_traj[idx][0], all_traj[idx][-1]]
        )
        all_traj_bar[0] = traj_bar

        for it in tqdm(range(self.nb_it)):
            # Start by computing the coefficients alpha_i^{f,f'} using DTW
            A = np.zeros((N, self.size_mean_traj, F))
            for i in range(N):
                # For each training trajectory, we compute the DTW matching between
                # the ith trajectory and the current mean one.
                dist, matching = r_fastdtw(all_traj[i], traj_bar)
                # We can then compute the coefficients alpha_i^{f,f'}
                A[i] = self.get_coeff_from_DTW(matching, F)

            # Once we have the coefficients, we can update the mean trajectory
            # For this, we use a Riemannian gradient descent algorithm
            cost_value = 0
            for f in range(self.size_mean_traj):
                cost, riemannian_gradient = self.create_cost(all_traj, A, f)
                optimizer = self.optimizer

                problem = pymanopt.Problem(
                    self.manifold, cost, riemannian_gradient=riemannian_gradient
                )
                pb_solved = optimizer.run(
                    problem=problem, initial_point=all_traj_bar[it, f]
                )
                # We can update the mean trajectory
                traj_bar[f] = pb_solved.point
                cost_value += pb_solved.cost
            all_traj_bar[it + 1] = traj_bar
            loss.append(cost_value)

            # We have two stopping criteria :
            # - We have reached a a priori fixed number of iterations
            # - The norm of the difference of two successive mean trajectories is smaller than a fixed threshold.
            if (
                np.max(np.linalg.norm(traj_bar - all_traj_bar[it], axis=(-2, -1)))
                < self.eps
            ):
                stop_crit = "conv"
                # print("Stopped because of convergence at iteration " + str(it + 1))
                break

        if it == self.nb_it - 1:
            stop_crit = "it"
            # print("Stopped because of run out of iterations")

        return traj_bar, loss, stop_crit

    def fit(self, X, y):
        """Fit the model

        Parameters
        ----------
        X : ndarray, shape (n_EEG, n_channels, size_EEG)
            The raw EEG on which we want to fit the classifier
        y : ndarray, shape (n_trajectories,)
            Labels for each EEG.

        Returns
        -------
        self : DTW_MDM instance
            The DTW_MDM instance.
        """

        # We start by getting the different classes
        self.classes_ = np.unique(y)

        # Then we preprocess the data to create the trajectories of covariances matrices
        cov_traj = self.preprocess_data(X, y)
        # print("Preprocessing done !")
        # We can then compute the mean trajectory for each class
        #print(cov_traj.shape)
        results = Parallel(n_jobs=-1)(
            delayed(self.compute_mean_traj_DTW)(cov_traj[y == ll])
            for ll in self.classes_
        )

        # We can then treat the different outputs of self.compute_mean_traj_DTW
        self.mean_traj = []
        self.loss = []
        self.stop_crit = []
        for i in range(len(self.classes_)):
            self.mean_traj.append(results[i][0])
            self.loss.append(results[i][1])
            self.stop_crit.append(results[i][2])

        return self

    def predict(self, X):
        """
         Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_EEG, n_channels, size_EEG)
            Set of EEG.

        Returns
        -------
        pred : ndarray, shape (n_matrices,)
            Predictions for each EEG according to the closest mean trajectory.
        """

        # We start by preprocessing the raw EEg to create the trajectories
        cov_traj = self.preprocess_data(X)

        # For each trajectory, we compute the Riemannian DTW distance between the trajectory
        # and the mean trajectory of each class and then select the smallest distance
        nb_traj = cov_traj.shape[0]
        nb_classes = len(self.classes_)
        label = []
        for i in range(nb_traj):
            dist = [
                r_fastdtw(cov_traj[i], self.mean_traj[j])[0] for j in range(nb_classes)
            ]
            label.append(self.classes_[np.argmin(dist)])
        return label

    def predict_proba(self, X):
        """
         Probability estimates.

        Parameters
        ----------
        X : ndarray, shape (n_EEG, n_channels, size_EEG)
            Set of EEG.

        Returns
        -------
        T : ndarray, shape (n_matrices,)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """

        # We start by preprocessing the raw EEg to create the trajectories
        cov_traj = self.preprocess_data(X)

        # For each trajectory, we compute the Riemannian DTW distance between the trajectory
        # and the mean trajectory of each class and then select the smallest distance
        nb_traj = cov_traj.shape[0]
        nb_classes = len(self.classes_)
        proba = np.zeros((nb_traj, nb_classes))
        dist = np.zeros((nb_traj,nb_classes))
        for i in range(nb_traj):
            dist[i] = [
                        r_fastdtw(cov_traj[i], self.mean_traj[j])[0] for j in range(nb_classes)
                    ]
        return softmax(-np.array(dist) ** 2)



class PT_MDM(BaseEstimator, ClassifierMixin):
    """Classification using the Riemannian mean to compute the mean trajectory

    Classification of covariances trajectories. First, the trajectories are created by cutting the original time series in smaller windows of size size_window.
    One covariance matrix is computed per window to form a trajectory. Then, for each class, the mean trajectory is computed by averaging each point
    using the Riemannain mean. Then, the classification uses a minimum distance to mean to affect the new trajectory
    to the closest average.

    Parameters
    ------------
    manifold : pymanopt.manifold
        The manifold on which the data lies on
    size_window : int
        The size of the windows used to create the n_trajectories
    cov_estimator : str, default = "scm"
        The covariance estimator used to estimate the covariance of each window on the trajectories. See pyriemann's doc
    """
    def __init__(self, manifold,size_window,cov_estimator):
        """Init.
        """
        self.manifold = manifold
        self.size_window = size_window
        self.cov_estimator = cov_estimator
        self.fgda = FGDA()

    def get_temporal_trajectories(self, X, width_bins, overlap):
        """ Creates the subwindows of X in order to compute the covariance trajectories.

        Parameters
        -------------
        X : array_like of size (n_channels, N)
            The original time series
        width_bins : int
            The size of the windows that will be extracted from X
        overlap : int
            The overlap that will be used to create the windows

        Returns
        -------------
        traj : array_like of size (n_windows, n_channels, width_bins)
            The array containing all the windows extracted

        """
        traj = []
        t = 0
        N = X.shape[1]
        while t < N:
            traj.append(X[:, t : (t + width_bins)])
            t += width_bins - overlap
        #return traj
        return np.array(traj[:-1])

    def preprocess_data(self, X, y=None):
        """ Preprocess the data in order to create the trajectories of covariance matrices.

        Parameters
        -------------
        X : array_like of size (n_series, n_channels, length_series)
            The original time series
        y : array_like of size (n_series,)
            The labels of each trajectory of X

        Returns
        -------------
        all_traj_filtred : array_like of size (n_series, length_traj, n_channels, n_channels)
            The array containing all the covariance matrix trajectories filtred using a FGDA filter.
        """

        # We start by extracting all the subwindows of the main time series X to create the trajectories
        traj_temp = []
        for i in range(X.shape[0]):
            traj_temp.append(self.get_temporal_trajectories(X[i], self.size_window, self.overlap))
        traj_temp = np.array(traj_temp)

        # Once the subwindows are extracted, we can compute one covariance matrix per sub window
        # therefore creating the covariance matrix trajectories.
        (nb_traj, length_traj, nb_sensors, length_EGG) = traj_temp.shape
        # print(length_traj)
        all_traj = np.zeros((nb_traj, length_traj, nb_sensors, nb_sensors))
        for j in range(length_traj):
            all_traj[:, j] = Covariances(estimator=self.cov_estimator).transform(
                traj_temp[:, j, :, :]
            )

        # Then we filter the trajectories
        if y is None:
            # If y is None, then the estimator (therefore the FGDA filter) has already been fitted,
            # so we just need to transform the data
            all_traj_filtred = self.fgda.transform(
                np.reshape(
                    all_traj,
                    (
                        all_traj.shape[0] * all_traj.shape[1],
                        all_traj.shape[2],
                        all_traj.shape[2],
                    ),
                )
            )
            all_traj_filtred = np.reshape(
                all_traj_filtred,
                (
                    all_traj.shape[0],
                    all_traj.shape[1],
                    all_traj.shape[2],
                    all_traj.shape[2],
                ),
            )
        else:
            # Otherwise, we need to fit the fgda filter using all the covariance matrices we just computed
            big_y = np.array(
                [y[i] for i in range(len(y)) for j in range(all_traj.shape[1])]
            )
            all_traj_filtred = self.fgda.fit_transform(
                np.reshape(
                    all_traj,
                    (
                        all_traj.shape[0] * all_traj.shape[1],
                        all_traj.shape[2],
                        all_traj.shape[2],
                    ),
                ),
                big_y,
            )
            all_traj_filtred = np.reshape(
                all_traj_filtred,
                (
                    all_traj.shape[0],
                    all_traj.shape[1],
                    all_traj.shape[2],
                    all_traj.shape[2],
                ),
            )

        return all_traj_filtred

    def compute_mean_traj(self, X):
        """Compute the mean trajectory by averaging each point using the Riemannain mean.

        Parameters
        ----------
        X : ndarray, shape (n_trajectories, n_matrices, n_channels, n_channels)
            Set of trajectories of SPD matrices.

        Returns
        -------
        mean_traj : ndarray, shape (n_matrices, n_channels, n_channels)
            The mean trajectory.
        """
        n = X.shape[1]
        mean_traj = np.array([mean_riemann(X[:, k]) for k in range(n)])
        return mean_traj


    def fit(self, X, y):
        """Fit the model

        Parameters
        ----------
        X : ndarray, shape (n_trajectories, n_channels, size_EEG)
            The raw EEG on which we want to fit the classifier
        y : ndarray, shape (n_trajectories,)
            Labels for each trajectory.

        Returns
        -------
        self : DTW_MDM instance
            The DTW_MDM instance.
        """

        # We start by getting the different classes
        self.classes_ = np.unique(y)

        # Then we proprocess the data
        cov_traj = self.preprocess_data(X, y)

        # We can then compute the mean trajectory for each class
        self.mean_traj = Parallel(n_jobs=-1)(
            delayed(self.compute_mean_traj)(cov_traj[y == ll])
            for ll in self.classes_
        )

        return self


    def distance_traj(self, X, Y):
        """Compute the distance from a trajectory to another

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            First trajectory of SPD matrices.

        Y : ndarray, shape (n_matrices, n_channels, n_channels)
            Second trajectory of SPD matrices.

        Returns
        -------
        d : int
            The distance between the first and the second trajectory.

        """
        n = len(X)
        d = np.sum([distance_riemann(X[i], Y[i]) ** 2 for i in range(n)])
        return d

    def predict(self, X):
        """
         Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trajectories, n_matrices, n_channels, n_channels)
            Set of trajectories of SPD matrices.

        Returns
        -------
        pred : ndarray, shape (n_matrices,)
            Predictions for each trajectory according to the closest mean trajectory.
        """

        cov_traj = self.preprocess_data(X)

        nb_traj = cov_traj.shape[0]
        nb_classes = len(self.classes_)
        label = []
        for i in range(nb_traj):
            dist = [
                self.distance_traj(cov_traj[i], self.mean_traj[j]) for j in range(nb_classes)
            ]
            label.append(self.classes_[np.argmin(dist)])
        return label


    def predict_proba(self, X):
        """
         Probability estimates.

        Parameters
        ----------
        X : ndarray, shape (n_EEG, n_channels, size_EEG)
            Set of EEG.

        Returns
        -------
        T : ndarray, shape (n_matrices,)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """

        # We start by preprocessing the raw EEg to create the trajectories
        cov_traj = self.preprocess_data(X)

        # For each trajectory, we compute the Riemannian DTW distance between the trajectory
        # and the mean trajectory of each class and then select the smallest distance
        nb_traj = cov_traj.shape[0]
        nb_classes = len(self.classes_)
        proba = np.zeros((nb_traj, nb_classes))
        dist = np.zeros((nb_traj,nb_classes))
        for i in range(nb_traj):
            dist[i] = [
                self.distance_traj(cov_traj[i], self.mean_traj[j]) for j in range(nb_classes)
            ]
        return softmax(-np.array(dist) ** 2)
