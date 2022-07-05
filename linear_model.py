import numpy as np
from scipy.special import expit
import time


class LinearModel:
    def __init__(
        self,
        loss_function,
        batch_size=None,
        step_alpha=1,
        step_beta=0, 
        tolerance=1e-5,
        max_iter=1000,
        random_seed=153,
        **kwargs
    ):
        """
        Parameters
        ----------
        loss_function : BaseLoss inherited instance
            Loss function to use
        batch_size : int
        step_alpha : float
        step_beta : float
            step_alpha and step_beta define the learning rate behaviour
        tolerance : float
            Tolerace for stop criterio.
        max_iter : int
            Max amount of epoches in method.
        """
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed

    def fit(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, training set.
        y : numpy.ndarray
            1d vector, target values.
        w_0 : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        trace : bool
            If True need to calculate metrics on each iteration.
        X_val : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y_val: numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : dict
            Keys are 'time', 'func', 'func_val'.
            Each key correspond to list of metric values after each training epoch.
        """
        
        N = X.shape[0]
        prev_w = None
        if w_0 is not None:
            self.w = np.copy(w_0)
        else:
            self.w = np.zeros(X.shape[1])
        
        if trace:
            if X_val is not None and y_val is not None:
                history = {'time': [], 'func': [], 'func_val': []}
            else:
                history = {'time': [], 'func': []}
        
        if self.batch_size is not None:
            num_of_iters = int(N // self.batch_size)
        for epoch in range(1, self.max_iter + 1):
            rand_perm = np.random.permutation(N)
            prev_w = np.copy(self.w)
            etha = self.step_alpha / (epoch ** self.step_beta)
            if trace:
                start_time = time.time()

            if self.batch_size is not None:
                for num_of_batch in range(num_of_iters):
                    X_batch = X[rand_perm[num_of_batch * self.batch_size:(num_of_batch + 1) * self.batch_size]]
                    y_batch = y[rand_perm[num_of_batch * self.batch_size:(num_of_batch + 1) * self.batch_size]]
                    self.w = self.w - etha * self.loss_function.grad(X_batch, y_batch, self.w)
            else:
                self.w = self.w - etha * self.loss_function.grad(X, y, self.w)

            if trace:
                history['time'].append(time.time() - start_time)
                history['func'].append(self.loss_function.func(X, y, self.w))
                if X_val is not None and y_val is not None:
                    history['func_val'].append(self.loss_function.func(X_val, y_val, self.w))

            if np.linalg.norm(prev_w - self.w) < self.tolerance:
                break
        if trace:
            return history
        
    def predict(self, X, threshold=0):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, test set.
        threshold : float
            Chosen target binarization threshold.

        Returns
        -------
        : numpy.ndarray
            answers on a test set
        """
        y_pred = np.empty(X.shape[0])
        weights = self.get_weights()
        scores = X.dot(weights)
        mask = (scores >= threshold)
        y_pred[mask] = 1
        y_pred[~mask] = -1
        
        return y_pred
