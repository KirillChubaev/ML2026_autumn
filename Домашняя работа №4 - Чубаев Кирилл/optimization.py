import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict


class LineSearchTool(object):
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
      if self._method == 'Constant':
          return self.c
      if self._method == 'Wolfe':
          result = scipy.optimize.line_search(
            oracle.func, oracle.grad, x_k, d_k,
            c1=self.c1, c2=self.c2
          )
          alpha = result.alpha if hasattr(result, 'alpha') else result[0]
          if alpha is not None:
            return alpha
      # Сюда попадаем при выборе Armijo или если Wolfe не нашёл шаг
      alpha = self.alpha_0 if previous_alpha is None else previous_alpha
      phi_0 = oracle.func_directional(x_k, d_k, 0)
      slope_0 = oracle.grad_directional(x_k, d_k, 0)

      while oracle.func_directional(x_k, d_k, alpha) > phi_0 + self.c1 * alpha * slope_0:
        alpha /= 2
      return alpha
      return None


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    if line_search_options is None:
        line_search_options = {}
    line_search_tool = LineSearchTool(**line_search_options)
    x_k = np.copy(x_0)

    start_time = datetime.now()
    previous_alpha = None

    for k in range(max_iter + 1):
        if not np.all(np.isfinite(x_k)):
            return x_k, 'computational_error', history

        value = oracle.func(x_k)
        grad = oracle.grad(x_k)
        if value is None or grad is None:
            return x_k, 'computational_error', history

        grad_norm = np.linalg.norm(grad)
        if not np.all(np.isfinite(value)) or not np.isfinite(grad_norm):
            return x_k, 'computational_error', history

        # Порог вычисляем один раз по градиенту в начальной точке.
        if k == 0:
            stop_norm = np.sqrt(tolerance) * grad_norm

        if trace:
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['func'].append(value)
            history['grad_norm'].append(grad_norm)
            if x_k.size <= 2:
                history['x'].append(x_k.copy())

        if display:
            print(f'k = {k}, f = {value}, grad_norm = {grad_norm:.6g}')

        if grad_norm <= stop_norm:
            return x_k, 'success', history
        if k == max_iter:
            break

        # Направление задаёт антиградиент, длину шага выбирает line_search
        direction = -grad
        alpha = line_search_tool.line_search(oracle, x_k, direction, previous_alpha)
        if alpha is None or not np.isfinite(alpha):
            return x_k, 'computational_error', history

        x_k = x_k + alpha * direction
        previous_alpha = 2 * alpha

    return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    if line_search_options is None:
        line_search_options = {}
    line_search_tool = LineSearchTool(**line_search_options)
    x_k = np.copy(x_0)
    start_time = datetime.now()

    for k in range(max_iter + 1):
        if not np.all(np.isfinite(x_k)):
            return x_k, 'computational_error', history

        value = oracle.func(x_k)
        grad = oracle.grad(x_k)
        if value is None or grad is None:
            return x_k, 'computational_error', history

        grad_norm = np.linalg.norm(grad)
        if not np.all(np.isfinite(value)) or not np.isfinite(grad_norm):
            return x_k, 'computational_error', history

        if k == 0:
            stop_norm = np.sqrt(tolerance) * grad_norm

        if trace:
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['func'].append(value)
            history['grad_norm'].append(grad_norm)
            if x_k.size <= 2:
                history['x'].append(x_k.copy())

        if display:
            print(f'k = {k}, f = {value}, grad_norm = {grad_norm:.6g}')

        if grad_norm <= stop_norm:
            return x_k, 'success', history
        if k == max_iter:
            break

        # Вычисляем гессиан в текущей точке.
        hess = oracle.hess(x_k)
        if scipy.sparse.issparse(hess):
            hess = hess.toarray()
        if hess is None or not np.all(np.isfinite(hess)):
            return x_k, 'computational_error', history

        # Решаем систему H d = -grad через разложение Холецкого.
        try:
            factor = scipy.linalg.cho_factor(hess, lower=True, check_finite=False)
            direction = scipy.linalg.cho_solve(factor, -grad, check_finite=False)
        except LinAlgError:
            return x_k, 'newton_direction_error', history

        if not np.all(np.isfinite(direction)):
            return x_k, 'computational_error', history

        # На каждой итерации начинаем подбор шага с единицы.
        alpha = line_search_tool.line_search(oracle, x_k, direction, previous_alpha=1.0)
        if alpha is None or not np.isfinite(alpha):
            return x_k, 'computational_error', history

        x_k = x_k + alpha * direction

    return x_k, 'iterations_exceeded', history