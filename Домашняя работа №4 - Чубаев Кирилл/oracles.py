import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Parameters
    ----------
        A : np.array or scipy.sparse.csr_matrix of shape (m, n)
            Feature matrix (training sample).
        b : np.array of shape (m,)
            Labels vector with values in {-1, 1}.
        regcoef : float
            Regularization coefficient.
    """
    def __init__(self, A, b, regcoef):
        self.A = A
        self.b = b
        self.regcoef = regcoef
        self._m = A.shape[0]

    def func(self, x):
        margins = self.b * (self.A @ x)
        return np.logaddexp(0, -margins).mean() + 0.5 * self.regcoef * np.dot(x, x)

    def grad(self, x):
        margins = self.b * (self.A @ x)
        weights = self.b * expit(-margins)
        return -(self.A.T @ weights) / self._m + self.regcoef * x

    def hess(self, x):
        margins = self.b * (self.A @ x)
        curvature = expit(margins) * expit(-margins)
        if scipy.sparse.issparse(self.A):
            weighted_A = self.A.multiply(curvature[:, None])
            hessian = (self.A.T @ weighted_A).toarray()
        else:
            weighted_A = self.A * curvature[:, None]
            hessian = self.A.T @ weighted_A
        hessian /= self._m
        hessian[np.diag_indices_from(hessian)] += self.regcoef
        return hessian


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    x = np.asarray(x, dtype=float)
    value = np.asarray(func(x)).item()
    grad = np.empty(x.size)
    for i in range(x.size):
        shifted = x.copy()
        shifted[i] += eps
        grad[i] = (np.asarray(func(shifted)).item() - value) / eps
    return grad


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    x = np.asarray(x, dtype=float)
    value = np.asarray(func(x)).item()
    single = np.empty(x.size)
    for i in range(x.size):
        shifted = x.copy()
        shifted[i] += eps
        single[i] = np.asarray(func(shifted)).item()

    hessian = np.empty((x.size, x.size))
    for i in range(x.size):
        shifted_i = x.copy()
        shifted_i[i] += eps
        for j in range(i, x.size):
            shifted_ij = shifted_i.copy()
            shifted_ij[j] += eps
            both = np.asarray(func(shifted_ij)).item()
            entry = ((both - single[i]) - (single[j] - value)) / eps**2
            hessian[i, j] = hessian[j, i] = entry
    return hessian
