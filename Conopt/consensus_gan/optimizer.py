import tensorflow as tf
from tensorflow.contrib.graph_editor import graph_replace


# Simultaneous gradient steps
class SimGDOptimizer(object):
    def __init__(self, learning_rate):
        self._sgd = tf.train.RMSPropOptimizer(learning_rate)

    def conciliate(self, d_loss, g_loss, d_vars, g_vars, global_step=None):
        # Compute gradients
        d_grads = tf.gradients(d_loss, d_vars)
        g_grads = tf.gradients(g_loss, g_vars)

        # Merge variable and gradient lists
        variables = d_vars + g_vars
        grads = d_grads + g_grads

        # Gradient updates
        reg_grads = list(zip(grads, variables))

        train_op = self._sgd.apply_gradients(reg_grads)

        return [train_op]


# Alternating gradient steps
class AltGDOptimizer(object):
    def __init__(self, learning_rate, d_steps=1, g_steps=1):
        self._d_sgd = tf.train.RMSPropOptimizer(learning_rate)
        self._g_sgd = tf.train.RMSPropOptimizer(learning_rate)
        self._d_steps = d_steps
        self._g_steps = g_steps

    def conciliate(self, d_loss, g_loss, d_vars, g_vars, global_step=None):
        d_steps = self._d_steps
        g_steps = self._g_steps

        # Select train_op
        train_op_g = self._g_sgd.minimize(g_loss, var_list=g_vars)
        train_op_d = self._d_sgd.minimize(d_loss, var_list=d_vars)

        return [train_op_d] * d_steps + [train_op_g] * g_steps


# Consensus optimization, method presented in the paper
class ConsensusOptimizer(object):
    def __init__(self, learning_rate, alpha=0.1, beta=0.9, eps=1e-8):
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate)
        self._eps = eps
        self._alpha = alpha
        self._beta = beta

    def conciliate(self, d_loss, g_loss, d_vars, g_vars, global_step=None):
        alpha = self._alpha
        beta = self._beta

        # Compute gradients
        d_grads = tf.gradients(d_loss, d_vars)
        g_grads = tf.gradients(g_loss, g_vars)

        # Merge variable and gradient lists
        variables = d_vars + g_vars
        grads = d_grads + g_grads

        # Reguliarizer
        reg = 0.5 * sum(
            tf.reduce_sum(tf.square(g)) for g in grads
        )
        # Jacobian times gradiant
        Jgrads = tf.gradients(reg, variables)

        # Gradient updates
        apply_vec = [
             (g + self._alpha * Jg, v)
             for (g, Jg, v) in zip(grads, Jgrads, variables) if Jg is not None
        ]

        train_op = self.optimizer.apply_gradients(apply_vec)

        return [train_op]


# Try to stabilize training by gradient clipping (suggested by reviewer)
class ClipOptimizer(object):
    def __init__(self, learning_rate, alpha=0.1, beta=0.9, eps=1e-8):
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate)
        self._eps = eps
        self._alpha = alpha
        self._beta = beta

    def conciliate(self, d_loss, g_loss, d_vars, g_vars, global_step=None):
        alpha = self._alpha
        beta = self._beta
        # Compute gradients

        d_grads = tf.gradients(d_loss, d_vars)
        g_grads = tf.gradients(g_loss, g_vars)
        # Merge variable and gradient lists
        d_grads, _ = tf.clip_by_global_norm(d_grads, self._alpha)
        g_grads, _ = tf.clip_by_global_norm(g_grads, self._alpha)

        variables = d_vars + g_vars
        grads = d_grads + g_grads

        # Jacobian times gradiant
        # Gradient updates
        apply_vec = list(zip(grads, variables))

        train_op = self.optimizer.apply_gradients(apply_vec)

        return [train_op]


# Like ConsensusOptimizer, but only take regularizer for discriminator (suggested by reviewer)
class SmoothingOptimizer(object):
    def __init__(self, learning_rate, alpha=0.1, beta=0.9, eps=1e-8):
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate)
        self._eps = eps
        self._alpha = alpha
        self._beta = beta

    def conciliate(self, d_loss, g_loss, d_vars, g_vars, global_step=None):
        alpha = self._alpha
        beta = self._beta
        # Compute gradients

        d_grads = tf.gradients(d_loss, d_vars)
        g_grads = tf.gradients(g_loss, g_vars)
        # Merge variable and gradient lists
        variables = d_vars + g_vars
        grads = d_grads + g_grads

        # Reguliarizer
        reg = 0.5 * sum(
            tf.reduce_sum(tf.square(g))  for g in g_grads
        )

        # Jacobian times gradiant
        Jgrads = tf.gradients(reg, d_vars)
        # Gradient updates
        apply_vec = [
             (g + self._alpha * Jg, v)
             for (g, Jg, v) in zip(d_grads, Jgrads, d_vars) if Jg is not None
        ]
        apply_vec += list(zip(g_grads, g_vars))

        train_op = self.optimizer.apply_gradients(apply_vec)

        return [train_op]

"""
class CESPOptimizer(object):
    def __init__(self, learning_rate, rho=[1,1]):
        self._sgd = tf.train.RMSPropOptimizer(learning_rate)
        self.rho = rho

    def indicator_x(self, l_x):
        if l_x < 0:
            return 1
        else:
            return 0
    
    def indicator_y(self, l_y):
        if l_y > 0:
            return 1
        else:
            return 0
        
    def sgn(self, x):
        if x > 0:
            return 1
        else:
            return -1

    def v_z(self, d_loss, g_loss, d_vars, g_vars):
        i, j = len(g_var), len(d_var)
        Hessian = tf.hessians(g_var, d_var)
        eigenvalue, eigenvector = tf.linalg.eigh(Hessian)
        d_grads = tf.gradients(d_loss, d_vars)
        g_grads = tf.gradients(g_loss, g_vars)
        g_eigenvalue, d_eigenvalue = eigenvalue[:i], eigenvalue[i:]
        g_eigenvector, d_eigenvector = eigenvector[:i], eigenvector[i:]
        v_z_m = []
        v_z_p = []
        for ii in range(i):
            v_z_m.append(self.indicator_x(g_eigenvalue[ii])*g_eigenvalue[ii]/(2*self.rho[0])*self.sgn(np.dot(g_eigenvector[ii].T, g_grads))*g_eigenvector[ii])
        for jj in range(j):
            v_z_p.append(self.indicator_y(d_eigenvalue[jj])*d_eigenvalue[jj]/(2*self.rho[1])*self.sgn(np.dot(d_eigenvector[jj].T, d_grads))*d_eigenvector[jj])
        return np.array([v])


    def conciliate(self, d_loss, g_loss, d_vars, g_vars, global_step=None):
        # Compute gradients
        d_grads = tf.gradients(d_loss, d_vars)
        g_grads = tf.gradients(g_loss, g_vars)

        # Merge variable and gradient lists
        variables = d_vars + g_vars
        grads = d_grads + g_grads

        # Gradient updates
        reg_grads = list(zip(grads, variables))

        train_op = self._sgd.apply_gradients(reg_grads)

        return [train_op]
        """