import tensorflow.compat.v1 as tf
from tensorflow.python.ops.gradients_impl import _hessian_vector_product
import numpy as np

# Idea
def ConOpt_with_CESP(G_loss, D_loss, learning_rate, gamma, alpha):

    optimizer = tf.train.AdagradOptimizer(learning_rate)
    train_var = tf.trainable_variables()

    # variables 
    G_var = [var for var in train_var if var.name.startswith('generator')]
    D_var = [var for var in train_var if var.name.startswith('discriminator')]

    #gradients
    d_grads = tf.gradients(D_loss, D_var)
    g_grads = tf.gradients(G_loss, G_var)

    # Merge variables and gradient lists
    variables = D_var + G_var
    grads = d_grads + g_grads

    # reguliarizer  L=1/2*norm(v)^2
    reg = 0.5*sum(tf.reduce_sum(tf.square(g)) for g in grads)

    # Jacobian times gradient       nabla L
    Jgrads_d = tf.gradients(reg, D_var)     # par(L)/par(theta)
    Jgrads_g = tf.gradients(reg, G_var)     # par(L)/par(phi)

    # eigenvalue and eigenvector
    v_g_min, e_g_min = get_min_eigvec(G_loss, G_var)
    v_d_max, e_d_max = get_min_eigvec(D_loss, D_var)

    # negative curvature
    nc_step_g = nc_step(v_g_min, e_g_min, G_var, alpha)
    nc_step_d = nc_step(v_d_max, e_d_max, D_var, alpha)

    # Gradient update
    grads_and_vars_discriminator = [
        (g+alpha*nc+gamma*Jg, v)    # grad+nc+gamma nabla L
        for (g,nc,Jg,v) in zip(d_grads, nc_step_d, Jgrads_d, D_var) if nc is not None and Jg is not None
    ]
    
    grads_and_vars_generator = [
        (g+alpha*nc+gamma*Jg, v)
        for (g,nc,Jg,v) in zip(g_grads, nc_step_g, Jgrads_g, G_var) if nc is not None and Jg is not None
    ]

    # Optimize
    G_optimizer = optimizer.minimize(G_loss, var_list=grads_and_vars_generator)
    D_optimizer = optimizer.minimize(D_loss, var_list=grads_and_vars_discriminator)
    return G_optimizer, D_optimizer, d_grads, g_grads, nc_step_d, nc_step_g, e_d_max, e_g_min

# Consensus Optimization
def ConOpt(G_loss, D_loss, learning_rate, gamma):

    optimizer = tf.train.AdagradOptimizer(learning_rate)
    train_var = tf.trainable_variables()
    
    # variables 
    G_var = [var for var in train_var if var.name.startswith('generator')]
    D_var = [var for var in train_var if var.name.startswith('discriminator')]
    
    #gradients
    d_grads = tf.gradients(D_loss, D_var)
    g_grads = tf.gradients(G_loss, G_var)
    
    # Merge variables and gradient lists
    variables = D_var + G_var
    grads = d_grads + g_grads
    
    # eigenvalue and eigenvector
    v_g_min, e_g_min = get_min_eigvec(G_loss, G_var)
    v_d_max, e_d_max = get_min_eigvec(D_loss, D_var)

    # reguliarizer  L=1/2*norm(v)^2
    reg = 0.5*sum(tf.reduce_sum(tf.square(g)) for g in grads)

    # Jacobian times gradient       nabla L
    Jgrads_d = tf.gradients(reg, D_var)     # par(L)/par(theta)
    Jgrads_g = tf.gradients(reg, G_var)     # par(L)/par(phi)
    
    # Gradient update
    grads_and_vars_discriminator = [
        (g+gamma*Jg, v)    # grad+gamma nabla L
        for (g,Jg,v) in zip(d_grads, Jgrads_d, D_var) if Jg is not None
    ]
    #print(vec_d, '=============================================================================================')
    grads_and_vars_generator = [
        (g+gamma*Jg, v)
        for (g,Jg,v) in zip(g_grads, Jgrads_g, G_var) if Jg is not None
    ]

    # Optimize
    G_optimizer = optimizer.minimize(G_loss, var_list=grads_and_vars_generator)
    D_optimizer = optimizer.minimize(D_loss, var_list=grads_and_vars_discriminator)
    return G_optimizer, D_optimizer, d_grads, g_grads, e_d_max, e_g_min

# Gradient
def Gradient(G_loss, D_loss, learning_rate):
    train_var = tf.trainable_variables()
    G_var = [var for var in train_var if var.name.startswith('generator')]
    D_var = [var for var in train_var if var.name.startswith('discriminator')]
    
    #gradients
    d_grads = tf.gradients(D_loss, D_var)
    g_grads = tf.gradients(G_loss, G_var)

    # eigenvalue and eigenvector
    v_g_min, e_g_min = get_min_eigvec(G_loss, G_var)
    v_d_max, e_d_max = get_min_eigvec(D_loss, D_var)

    # Optimize
    G_optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(G_loss, var_list=G_var)
    D_optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(D_loss, var_list=D_var)
    return G_optimizer, D_optimizer, d_grads, g_grads, e_d_max, e_g_min

# Negative Curvature Explioitation for Local Saddle Point Problem
def CESP(G_loss, D_loss, learning_rate, alpha):
    optimizer = tf.train.AdagradOptimizer(learning_rate)
    train_var = tf.trainable_variables()
    
    # variables 
    G_var = [var for var in train_var if var.name.startswith('generator')]
    D_var = [var for var in train_var if var.name.startswith('discriminator')]
    vars = D_var + G_var

    # gradients
    d_grads = tf.gradients(D_loss, D_var)
    g_grads = tf.gradients(G_loss, G_var)
    grads = d_grads+g_grads

    # eigenvalue and eigenvector
    v_g_min, e_g_min = get_min_eigvec(G_loss, G_var)
    v_d_max, e_d_max = get_min_eigvec(D_loss, D_var)

    # negative curvature
    nc_step_g = nc_step(v_g_min, e_g_min, G_var, alpha)
    nc_step_d = nc_step(v_d_max, e_d_max, D_var, alpha)

    # Gradient update
    grads_and_vars_discriminator = [
        (g+alpha*nc, v)    # grad+nc` 
        for (g,nc,v) in zip(d_grads, nc_step_d, D_var) if nc is not None
    ]
    
    grads_and_vars_generator = [
        (g+alpha*nc, v)
        for (g,nc,v) in zip(g_grads, nc_step_g, G_var) if nc is not None
    ]
    
    # Optimize
    G_optimizer = optimizer.minimize(G_loss, var_list=grads_and_vars_generator)
    D_optimizer = optimizer.minimize(D_loss, var_list=grads_and_vars_discriminator)
    return G_optimizer, D_optimizer, d_grads, g_grads, nc_step_d, nc_step_g, e_d_max, e_g_min


### Helper function
# For Negative Curvature Step
def nc_step(v, e, _vars, lr):
    d = lr* tf.abs(e) * v
    d_zero = tf.zeros_like(d)
    # If the minimum eigenvalue is < 0: then we do the update; otherwise just update with a zero vector
    assign_ops = tf.cond(e < 0, lambda: _apply_step(d, _vars), lambda: _apply_step(d_zero, _vars))
    return assign_ops

def get_min_eigvec(loss, vars):
    iterations = 10
    eps = 3
    v = [_get_initial_vector(vars)]
    eigvals = []
    grad = _list_to_tensor(tf.gradients(loss, vars))
    for i in range(iterations):
        # Power iteration with the shifted Hessian
        v_new = _list_to_tensor(_hessian_vector_product(loss, vars, _tensor_to_list(v[i], vars)))
        v.append(eps*v[i] - v_new)
        v[i+1] = _normalize(v[i+1])

        # Get corresponding eigenvalue
        eigval = tf.reduce_sum(tf.multiply(v[i], _list_to_tensor(
            _hessian_vector_product(loss, vars, _tensor_to_list(v[i], vars)))))
        eigvals.append(eigval)

    idx = iterations -1 #tf.cast(tf.argmin(eigvals[3:iterations-1]), tf.int32)
    e = tf.gather(eigvals, idx)
    v = tf.gather(v, idx)

    _sign = -tf.sign(tf.reduce_sum(tf.multiply(grad, v)))
    v *= _sign

    return v, e

def _get_initial_vector(_vars):
    v = tf.random_uniform([get_num_weights(_vars)])
    v = _normalize(v)
    return v

def _normalize(v):
    v /= tf.sqrt(tf.reduce_sum(tf.square(v), 0, keep_dims=True))
    return v

def _tensor_to_list(tensor, _vars):
    shape = [layer.get_shape().as_list() for layer in _vars]
    tensor_list = []
    offset = 0
    for sh in shape:
        end = offset + np.prod(sh)
        tensor_list.append(tf.reshape(tensor[offset:end], sh))
        offset = end
    return tensor_list

def _list_to_tensor(_list):
    _tensor = tf.concat([tf.reshape(layer, [-1]) for layer in _list], axis=0)
    return _tensor

def _apply_step(d, _vars):
    # Reshape d to fit the layer shapes
    shape = [layer.get_shape().as_list() for layer in _vars]
    d_list = []
    offset = 0
    for sh in shape:
        end = offset + np.prod(sh)
        d_list.append(tf.reshape(d[offset:end], sh))
        offset = end
    # Do the assign operations
    assign_ops = []
    for i in range(len(_vars)):
        assign_ops.append(tf.assign_add(_vars[i], d_list[i]))
    return assign_ops

def get_num_weights(params):
    shape = [layer.get_shape().as_list() for layer in params]
    n = 0
    for sh in shape:
        n_layer = 1
        for dim in sh:
            n_layer *= dim
        n += n_layer
    return n