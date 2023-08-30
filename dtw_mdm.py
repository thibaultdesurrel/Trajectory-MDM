import numpy as np
import matplotlib.pyplot as plt

import pymanopt.optimizers
import pymanopt.manifolds

from sklearn.base import BaseEstimator, ClassifierMixin

from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.base import logm
from pyriemann.utils.mean import mean_riemann

from tqdm import tqdm

from joblib import Parallel, delayed

from fastdtw import fastdtw

import scipy.stats as stat

from utils import *

class DTW_MDM(BaseEstimator, ClassifierMixin):
    """ Classification using the DTW to compute the mean trajectory

    Classification of covariances trajectories. For each class, the mean trajectory is computed using a iterative two step algorithm :
    First, a matching between each training trajectory and the current mean one is computed using DTW. Then, a weighted average is
    computed to update the mean trajectory. Finally, the classification uses a minimum DTW distance to mean to affect the new trajectory
    to the closest average.

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
    initial_traj : array_like of size (size_mean_traj, n_channels, n_channels)
        The initial mean trajectory
    optimizer : pymanopt.optimizer, default = pymanopt.optimizers.SteepestDescent
        The pymanopt optimizer used to solve the minimization problem
    """
    def __init__(
        self,
        manifold,
        nb_it,
        eps,
        size_mean_traj,
        initial_traj=np.array([0]),
        optimizer=pymanopt.optimizers.SteepestDescent(verbosity=0),
    ):
        """Init.
        """
        self.manifold = manifold
        self.nb_it = nb_it
        self.eps = eps
        self.size_mean_traj = size_mean_traj
        self.initial_traj = initial_traj
        self.optimizer = optimizer
        self.loss = None

    def random_trajectories(self, T, all_M):
        """ Generate a random trajectory of SPD matrices of size T.
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
        # We need to know in which subinterval lies each time t
        for t in all_t:
            i = 0
            while t > (i + 1) / (n_M - 1):
                i += 1
            # Then we can compute a SPD matrix on the geodesic between all_M[i] and all_M[i + 1]
            v = self.manifold.log(all_M[i], all_M[i + 1])
            # We add a small random noise
            epsilon = stat.norm.rvs(scale=0.5, size=n)
            t_tilde = (t - i / (n_M - 1)) * ((n_M - 1))
            traj.append(
                exp(all_M[i], t_tilde * v)
                + epsilon.reshape(n, 1) @ epsilon.reshape(1, n)
            )
        traj = np.array(traj)
        return traj

    def create_cost(self, all_traj, A, f):
        """ Create the cost function as well as its Riemannian gradient associated to our problem

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
            # The gradient is -2\sum_i \sum_{f'} Log_X(X_i^f')
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
        """ Get the coefficients alpha_i^{f,f'} from the DTW matching

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
        """ Compute the mean trajectory of the trajectories in all_traj

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

        # We start by initializing the mean trajectory to the geometric mean at each frequency
        loss = []
        if self.initial_traj.shape != (self.size_mean_traj, C, C):
            idx = np.random.randint(N)
            traj_bar = self.random_trajectories(
                self.size_mean_traj, [all_traj[idx][0], all_traj[idx][-1]]
            )
            all_traj_bar[0] = traj_bar
            """
            # If no initial trajectory is given, we initialize with the Riemanniann mean for each points.
            eucl_mean_traj = np.mean(all_traj, axis=0)
            # optimizer = SteepestDescent(verbosity=0)

            # cost_value = 0
            for f in range(F):
                cost, riemannian_gradient = self.create_cost(all_traj, A, f)
                optimizer = self.optimizer

                problem = pymanopt.Problem(
                    self.manifold, cost, riemannian_gradient=riemannian_gradient
                )
                pb_solved = optimizer.run(problem = problem, initial_point=eucl_mean_traj[f])
                traj_bar[f] = pb_solved.point
                # cost_value += pb_solved.cost
            all_traj_bar[0] = traj_bar
            # loss.append(cost_value)
            """
        else:
            # If an initial trajectory is given, we use it
            traj_bar = self.initial_traj
            all_traj_bar[0] = self.initial_traj

        for it in tqdm(range(self.nb_it)):
            # Start by computing the coefficients alpha_i^{f,f'} using DTW
            A = np.zeros((N, self.size_mean_traj, F))
            for i in range(N):
                dist, matching = r_fastdtw(all_traj[i], traj_bar)
                A[i] = self.get_coeff_from_DTW(matching, F)

            # Once we have the coefficients, we can update the mean trajectory
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
                traj_bar[f] = pb_solved.point
                cost_value += pb_solved.cost
            all_traj_bar[it + 1] = traj_bar
            loss.append(cost_value)

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
        X : ndarray, shape (n_trajectories, n_matrices, n_channels, n_channels)
            Set of trajectories of SPD matrices.
        y : ndarray, shape (n_trajectories,)
            Labels for each trajectory.

        Returns
        -------
        self : DTW_MDM instance
            The DTW_MDM instance.
        """

        # We start by getting the different classes
        self.classes = np.unique(y)

        # We can then compute the mean trajectory for each class
        results = Parallel(n_jobs = -1)(
            delayed(self.compute_mean_traj_DTW)(X[y == ll])
            for ll in self.classes
        )

        # We can then treat the different outputs of self.compute_mean_traj_DTW
        self.mean_traj = []
        self.loss = []
        self.stop_crit = []
        for i in range(len(self.classes)):
            self.mean_traj.append(results[i][0])
            self.loss.append(results[i][1])
            self.stop_crit.append(results[i][2])

        return self

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
        nb_traj = X.shape[0]
        nb_classes = len(self.classes)
        label = []
        for i in range(nb_traj):
            dist = [r_fastdtw(X[i], self.mean_traj[j])[0] for j in range(nb_classes)]
            label.append(self.classes[np.argmin(dist)])
        return label

    def plot_loss(self):
        """ Plot the loss of the training
        """
        if self.loss == None:
            raise Exception("You need to fit the classifier before plotting the loss.")
        plt.figure()
        for i in range(len(self.classes)):
            plt.plot(self.loss[i], label="Class " + str(self.classes[i]))
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Plot of the loss during the training")
        plt.legend()
        plt.show()


    def DTW_visu(self,X,Y,matching):
        axis_X = np.arange(len(X))
        axis_Y = np.linspace(0,len(X)-1,len(Y))

        for (x,y) in matching:
            plt.plot([axis_X[x],axis_Y[y]],[1,0], '--r',alpha = 0.05)

        plt.plot(axis_X,np.ones_like(axis_X),'o')
        plt.plot(axis_Y,np.zeros_like(axis_Y),'o')

        plt.ylim(-0.5,1.5)
        plt.xticks([])
        plt.yticks([])

    def all_DTW_visu(self, X):
        nb_traj = X.shape[0]
        plt.figure()
        nb_classes = len(self.classes)
        for i in range(nb_traj):
            all_dist = []
            all_matching = []
            for j in range(nb_classes):
                dist, matching = r_fastdtw(X[i], self.mean_traj[j])
                all_dist.append(dist)
                all_matching.append(matching)
            class_idx = np.argmin(all_dist)
            plt.subplot(nb_classes,1,class_idx+1)
            self.DTW_visu(X[i],self.mean_traj[class_idx],all_matching[class_idx])
            plt.title(str(self.classes[class_idx]))
        plt.show()

class DTW_MDM_E(BaseEstimator, ClassifierMixin):
    """ Classification using the Euclidian DTW to compute the mean trajectory

    Classification of covariances trajectories. For each class, the mean trajectory is computed using a iterative two step algorithm :
    First, a matching between each training trajectory and the current mean one is computed using DTW. Then, a weighted average is
    computed to update the mean trajectory. Finally, the classification uses a minimum DTW distance to mean to affect the new trajectory
    to the closest average.

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
    initial_traj : array_like of size (size_mean_traj, n_channels, n_channels)
        The initial mean trajectory
    optimizer : pymanopt.optimizer, default = pymanopt.optimizers.SteepestDescent
        The pymanopt optimizer used to solve the minimization problem
    """
    def __init__(
        self,
        nb_it,
        eps,
        size_mean_traj,
        optimizer=pymanopt.optimizers.SteepestDescent(verbosity=0),
    ):
        """Init.
        """
        self.nb_it = nb_it
        self.eps = eps
        self.size_mean_traj = size_mean_traj
        self.optimizer = optimizer
        self.loss = None

    def random_trajectories(self, T, all_M):
        """ Generate a random trajectory of SPD matrices of size T.
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
        # We need to know in which subinterval lies each time t
        for t in all_t:
            i = 0
            while t > (i + 1) / (n_M - 1):
                i += 1
            # Then we can compute a SPD matrix on the geodesic between all_M[i] and all_M[i + 1]
            v = self.manifold.log(all_M[i], all_M[i + 1])
            # We add a small random noise
            epsilon = stat.norm.rvs(scale=0.5, size=n)
            t_tilde = (t - i / (n_M - 1)) * ((n_M - 1))
            traj.append(
                exp(all_M[i], t_tilde * v)
                + epsilon.reshape(n, 1) @ epsilon.reshape(1, n)
            )
        traj = np.array(traj)
        return traj


    def get_coeff_from_DTW(self, matching, F):
        """ Get the coefficients alpha_i^{f,f'} from the DTW matching

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
        """ Compute the mean trajectory of the trajectories in all_traj

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

        # We start by initializing the mean trajectory to the euclidian mean at each frequency
        loss = []

        traj_bar = np.mean(all_traj, axis=0)
        all_traj_bar[0] = traj_bar

        for it in tqdm(range(self.nb_it)):
            # Start by computing the coefficients alpha_i^{f,f'} using DTW
            A = np.zeros((N, self.size_mean_traj, F))
            for i in range(N):
                dist, matching = fastdtw(all_traj[i], traj_bar, dist=dist_F)
                A[i] = self.get_coeff_from_DTW(matching, F)

            # Once we have the coefficients, we can update the mean trajectory
            cost_value = 0
            for f in range(self.size_mean_traj):
                traj_bar[f] = np.zeros_like(traj_bar[f])
                for i in range(N):
                    for f_prime in range(F):
                        traj_bar[f] += A[i,f,f_prime]*all_traj[i,f_prime]

                traj_bar[f] /= np.sum(A[:,f,:])
            all_traj_bar[it + 1] = traj_bar
            loss.append(cost_value)

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
        X : ndarray, shape (n_trajectories, n_matrices, n_channels, n_channels)
            Set of trajectories of SPD matrices.
        y : ndarray, shape (n_trajectories,)
            Labels for each trajectory.

        Returns
        -------
        self : DTW_MDM instance
            The DTW_MDM instance.
        """

        # We start by getting the different classes
        self.classes = np.unique(y)

        # We can then compute the mean trajectory for each class
        results = Parallel(n_jobs = -1)(
            delayed(self.compute_mean_traj_DTW)(X[y == ll])
            for ll in self.classes
        )

        # We can then treat the different outputs of self.compute_mean_traj_DTW
        self.mean_traj = []
        self.loss = []
        self.stop_crit = []
        for i in range(len(self.classes)):
            self.mean_traj.append(results[i][0])
            self.loss.append(results[i][1])
            self.stop_crit.append(results[i][2])

        return self

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
        nb_traj = X.shape[0]
        nb_classes = len(self.classes)
        label = []
        for i in range(nb_traj):
            dist = [fastdtw(X[i], self.mean_traj[j], dist=dist_F)[0] for j in range(nb_classes)]
            label.append(self.classes[np.argmin(dist)])
        return label


class traj_MDM_R(BaseEstimator, ClassifierMixin):
    def __init__(self, manifold):
        self.manifold = manifold

    def compute_mean_traj(self, X):
        n = X.shape[1]
        return np.array([mean_riemann(X[:, k]) for k in range(n)])

    def fit(self, X, y):
        """Fit the model
        Parameters
        ----------
        X : ndarray, shape (n_trajectories, n_matrices, n_channels, n_channels)
            Set of trajectories of SPD matrices.
        y : ndarray, shape (n_trajectories,)
            Labels for each trajectory.

        Returns
        -------
        self : traj_MDM instance
            The traj_MDM instance.
        """

        self.classes = np.unique(y)

        # self.mean_traj =  Parallel(n_jobs=-1)(
        #    delayed(self.compute_mean_traj_DTW)(X[y == ll],np.where(self.classes == ll)[0][0]) for ll in self.classes
        # )
        self.mean_traj = []
        for ll in self.classes:
            self.mean_traj.append(self.compute_mean_traj(X[y == ll]))
        # self.loss = results[1]
        return self

    def distance_traj(self, X, Y):
        n = len(X)
        return np.sum([distance_riemann(X[i], Y[i]) ** 2 for i in range(n)])

    def predict(self, X):
        """
         Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trajectories, n_matrices, n_channels, n_channels)
            Set of trajectories of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Predictions for each trajectory according to the closest mean trajectory.
        """
        nb_traj = X.shape[0]
        nb_classes = len(self.classes)
        label = []
        for i in range(nb_traj):
            dist = [
                self.distance_traj(X[i], self.mean_traj[j]) for j in range(nb_classes)
            ]
            label.append(self.classes[np.argmin(dist)])
        return label


class traj_MDM_E(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def compute_mean_traj(self, X):
        return np.mean(X, axis=0)

    def fit(self, X, y):
        """Fit the model
        Parameters
        ----------
        X : ndarray, shape (n_trajectories, n_matrices, n_channels, n_channels)
            Set of trajectories of SPD matrices.
        y : ndarray, shape (n_trajectories,)
            Labels for each trajectory.

        Returns
        -------
        self : traj_MDM instance
            The traj_MDM instance.
        """

        self.classes = np.unique(y)

        # self.mean_traj =  Parallel(n_jobs=-1)(
        #    delayed(self.compute_mean_traj_DTW)(X[y == ll],np.where(self.classes == ll)[0][0]) for ll in self.classes
        # )
        self.mean_traj = []
        for ll in self.classes:
            self.mean_traj.append(self.compute_mean_traj(X[y == ll]))
        # self.loss = results[1]
        return self

    def distance_traj(self, X, Y):
        n = len(X)
        return np.sum([np.linalg.norm(X[i] - Y[i]) ** 2 for i in range(n)])

    def predict(self, X):
        """
         Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trajectories, n_matrices, n_channels, n_channels)
            Set of trajectories of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Predictions for each trajectory according to the closest mean trajectory.
        """
        nb_traj = X.shape[0]
        nb_classes = len(self.classes)
        label = []
        for i in range(nb_traj):
            dist = [
                self.distance_traj(X[i], self.mean_traj[j]) for j in range(nb_classes)
            ]
            label.append(self.classes[np.argmin(dist)])
        return label
