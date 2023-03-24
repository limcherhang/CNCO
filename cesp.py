import logging
import torch
import time
from func import time_convertor

logger = logging.getLogger(__name__)

### Helper function
# For Negative Curvature Step

device = "cuda" if torch.cuda.is_available() else "cpu"


def nc_step_for_g(v, e, grad, lr):
    sign = torch.sign(v * grad)
    d = lr * torch.abs(e) * v * sign
    d_zero = torch.zeros_like(d)

    # If the minimum eigenvalue is < 0: then we do the update; otherwise just update with a zero vector
    # assign_ops = torch.where(e < 0, lambda: _apply_step(d, _vars), lambda: _apply_step(d_zero, _vars))
    if e < 0:
        assign_ops = d.to(device)
    else:
        assign_ops = d_zero.to(device)
    return assign_ops


def nc_step_for_d(v, e, grad, lr):
    sign = torch.sign(v * grad)
    d = lr * torch.abs(e) * v * sign
    d_zero = torch.zeros_like(d)

    # If the  maximum eigenvalue is > 0: then we do the update; otherwise just update with a zero vector
    # assign_ops = torch.where(e < 0, lambda: _apply_step(d, _vars), lambda: _apply_step(d_zero, _vars))
    if e > 0:
        assign_ops = d.to(device)
    else:
        assign_ops = d_zero.to(device)
    return assign_ops


def Hessian_vec_prod(flatten_grad, param, vecs):
    # vecs = _list_to_tensor(vecs).to(device)
    tensor_list = _list_to_tensor([i.mul(j) for i, j in zip(flatten_grad, vecs)])
    prod = torch.mean(tensor_list)
    grad = torch.autograd.grad(prod, param, retain_graph=True)

    return _list_to_tensor([g.detach() for g in grad]).to(device)


def get_min_eigvec(grad, param):
    iterations = 5
    eps = 3
    v = [_get_initial_vector(param)]
    eigvals = []

    flatten_grad = _list_to_tensor(grad).to(device)
    for i in range(iterations):
        v_new = Hessian_vec_prod(flatten_grad, param, v)

        v.append(eps * v[i] - v_new)
        v[i + 1] = _normalize(v[i + 1])

        # Get corresponding eigenvalue
        eigval = torch.sum(torch.multiply(v[i], v_new))
        eigvals.append(eigval)

    eigvalue = eigvals[iterations - 1]
    eigvector = v[iterations - 1]

    _sign = torch.sign(torch.multiply(flatten_grad, eigvector))

    eigvector *= _sign

    return eigvector, eigvalue, flatten_grad


def _get_initial_vector(_vars):
    v = torch.randn([get_num_weights(_vars)]).to(device)
    v = _normalize(v)
    return v


def _normalize(v):
    v /= torch.sqrt(torch.sum(torch.square(v), 0, keepdim=True))
    return v


def get_num_weights(params):
    shape = [list(layer.shape) for layer in params]
    n = 0
    for sh in shape:
        n_layer = 1
        for dim in sh:
            n_layer *= dim
        n += n_layer
        # print(n)
    return n


def get_num_weights_each_param(params):
    n = []
    n_each = []
    for i in range(len(params)):
        n.append(params[i].shape)
        if len(params[i].shape) == 1:
            n_each.append(params[i].shape[0])
        else:
            n_each.append(params[i].shape[0] * params[i].shape[1])
    return n, n_each


def _tensor_to_list(tensor):  # one-dim list
    tensor_list = []
    if len(tensor.shape) == 1:
        for i in range(tensor.shape[0]):
            tensor_list.append(tensor[i])
            # print(float(tensor[i]))
    else:
        for i in range(tensor.shape[0]):
            tmp = []
            for j in range(tensor.shape[1]):
                tmp.append(tensor[i][j])
            tensor_list += tmp
    return tensor_list


def _list_to_tensor(_list):  # n-dim list --> 1-dim tensor
    _tensor = torch.cat([torch.reshape(layer, [-1]) for layer in _list], axis=0)
    return _tensor.to(device)
