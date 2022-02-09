import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def SimGA(disc, gen, d_loss, g_loss, opt_gen, opt_disc, gamma, alpha, args):
    d_param = [d for d in disc.parameters()]                                     
    g_param = [g for g in gen.parameters()]

    #### Get grad()
    grad_d = torch.autograd.grad(d_loss, disc.parameters(), create_graph=True)
    grad_g = torch.autograd.grad(g_loss, gen.parameters(), create_graph=True)

    if args.is_eigen==1:
        v_g_min, e_g_min, flatten_grad_g = get_min_eigvec(grad_g, g_param)
        v_d_max, e_d_max, flatten_grad_d = get_min_eigvec(grad_d, d_param)      
        nc_step_g = nc_step_for_g(v_g_min, e_g_min, flatten_grad_g, alpha*10)
        nc_step_d = nc_step_for_d(v_d_max, e_d_max, flatten_grad_d, alpha)
    else:
        v_g_min = torch.zeros(get_num_weights(g_param))
        e_g_min = torch.tensor([0])
        v_d_max = torch.zeros(get_num_weights(g_param))
        e_d_max = torch.tensor([0])
        flatten_grad_d = _list_to_tensor(grad_d).to(device)
        flatten_grad_g = _list_to_tensor(grad_g).to(device)
        nc_step_g = v_g_min
        nc_step_d = v_d_max

    grad = grad_d+grad_g
    param = d_param+g_param

    reg = 0.0*sum(torch.sum(torch.square(g)) for g in grad)

    Jgrad = torch.autograd.grad(reg, param)
    
    Jgrad_d = Jgrad[0:len(grad_d)]
    Jgrad_g = Jgrad[len(grad_d):]
    del Jgrad

    final_grad_d = [g + 0 * gamma * j for j, g in zip(Jgrad_d, grad_d)]
    final_grad_g = [g + 0 * gamma * j for j, g in zip(Jgrad_g, grad_g)]

    # backward
    for p, g in zip(d_param, final_grad_d):                                  
        p.grad = g.detach()                                     

    for p, g in zip(g_param, final_grad_g):                                     
        p.grad = g.detach()                                     

    opt_disc.step()
    opt_gen.step()

    return opt_disc, opt_gen, nc_step_g, nc_step_d, e_d_max, e_g_min, flatten_grad_d, flatten_grad_g

def ConOpt(disc, gen, d_loss, g_loss, opt_gen, opt_disc, gamma, alpha, args):                  
    d_param = [d for d in disc.parameters()]                                     
    g_param = [g for g in gen.parameters()]                                     

    #### Get grad()
    grad_d = torch.autograd.grad(d_loss, disc.parameters(), retain_graph=True, create_graph=True)
    grad_g = torch.autograd.grad(g_loss, gen.parameters(), retain_graph=True, create_graph=True) 
    
    if args.is_eigen==1:
        v_g_min, e_g_min, flatten_grad_g = get_min_eigvec(grad_g, g_param)
        v_d_max, e_d_max, flatten_grad_d = get_min_eigvec(grad_d, d_param)      
        nc_step_g = nc_step_for_g(v_g_min, e_g_min, flatten_grad_g, alpha*10)
        nc_step_d = nc_step_for_d(v_d_max, e_d_max, flatten_grad_d, alpha)
    else:
        v_g_min = torch.zeros(get_num_weights(g_param))
        e_g_min = torch.tensor([0])
        v_d_max = torch.zeros(get_num_weights(g_param))
        e_d_max = torch.tensor([0])
        flatten_grad_d = _list_to_tensor(grad_d).to(device)
        flatten_grad_g = _list_to_tensor(grad_g).to(device)
        nc_step_g = v_g_min
        nc_step_d = v_d_max

    grad = grad_d+grad_g
    param = d_param+g_param

    # reg_d = 0.5*sum(torch.sum(g**2) for g in grad_d)
    # reg_g = 0.5*sum(torch.sum(g**2) for g in grad_g)
    # reg = 0.5*sum(torch.sum(g**2) for g in grad)
    reg = 0.5*sum(torch.sum(torch.square(g)) for g in grad)

    # Jgrad_d = torch.autograd.grad(reg_d, d_param)
    # Jgrad_g = torch.autograd.grad(reg_g, g_param)
    Jgrad = torch.autograd.grad(reg, param)
    
    Jgrad_d = Jgrad[0:len(grad_d)]
    Jgrad_g = Jgrad[len(grad_d):]
    del Jgrad

    final_grad_d = [g + gamma * j for j, g in zip(Jgrad_d, grad_d)]
    final_grad_g = [g + gamma * j for j, g in zip(Jgrad_g, grad_g)]

    for p,g in zip(disc.parameters(), final_grad_d):
        p.grad = g.detach()

    for p,g in zip(gen.parameters(), final_grad_g):
        p.grad = g.detach()

    opt_disc.step()
    opt_gen.step()                                    

    return opt_disc, opt_gen, nc_step_g, nc_step_d, e_d_max, e_g_min, flatten_grad_d, flatten_grad_g

def CESP(disc, gen, d_loss, g_loss, opt_gen, opt_disc, gamma, alpha, args):   
    d_param = [d for d in disc.parameters()]
    g_param = [g for g in gen.parameters()]

    size_d, size_each_d = get_num_weights_each_param(d_param)
    size_g, size_each_g = get_num_weights_each_param(g_param)

    #### Get grad()
    grad_d = torch.autograd.grad(d_loss, disc.parameters(), retain_graph=True, create_graph=True)
    grad_g = torch.autograd.grad(g_loss, gen.parameters(), retain_graph=True, create_graph=True)

    v_g_min, e_g_min, flatten_grad_g = get_min_eigvec(grad_g, g_param)
    v_d_max, e_d_max, flatten_grad_d = get_min_eigvec(grad_d, d_param)

    nc_step_g = nc_step_for_g(v_g_min, e_g_min, flatten_grad_g, alpha*10)
    nc_step_d = nc_step_for_d(v_d_max, e_d_max, flatten_grad_d, alpha)
    nc_step_g_new = []
    nc_step_d_new = []
    tmp_d = 0
    for i in range(len(grad_d)):
        nc_step_d_new.append(nc_step_d[tmp_d:size_each_d[i]+tmp_d].view(size_d[i]))
        tmp_d += size_each_d[i]
    tmp_g = 0
    for i in range(len(grad_g)):
        nc_step_g_new.append(nc_step_g[tmp_g:size_each_g[i]+tmp_g].view(size_g[i]))
        tmp_g += size_each_g[i]

    grad = grad_d+grad_g
    param = d_param+g_param

    reg = 0.0*sum(torch.sum(torch.square(g)) for g in grad)

    Jgrad = torch.autograd.grad(reg, param)
    
    Jgrad_d = Jgrad[0:len(grad_d)]
    Jgrad_g = Jgrad[len(grad_d):]
    del Jgrad

    final_grad_d = [g + nc for g, nc in zip(grad_d, nc_step_d_new)]
    final_grad_g = [g + nc for g, nc in zip(grad_g, nc_step_g_new)]

    # backward
    for p,g in zip(disc.parameters(), final_grad_d):
        p.grad = g.detach()
    for p,g in zip(gen.parameters(), final_grad_g):
        p.grad = g.detach()

    opt_disc.step()
    opt_gen.step()

    return opt_disc, opt_gen, nc_step_g, nc_step_d, e_d_max, e_g_min, flatten_grad_d, flatten_grad_g

def CNCO(disc, gen, d_loss, g_loss, opt_gen, opt_disc, gamma, alpha, args):
    d_param = [d for d in disc.parameters()]                                     
    g_param = [g for g in gen.parameters()]

    size_d, size_each_d = get_num_weights_each_param(d_param)
    size_g, size_each_g = get_num_weights_each_param(g_param)

    #### Get grad()
    grad_d = torch.autograd.grad(d_loss, disc.parameters(), retain_graph=True, create_graph=True)
    grad_g = torch.autograd.grad(g_loss, gen.parameters(), retain_graph=True, create_graph=True)                                                                      

    v_g_min, e_g_min, flatten_grad_g = get_min_eigvec(grad_g, g_param)
    v_d_max, e_d_max, flatten_grad_d = get_min_eigvec(grad_d, d_param)
    
    nc_step_g = nc_step_for_g(v_g_min, e_g_min, flatten_grad_g, alpha*10)
    nc_step_d = nc_step_for_d(v_d_max, e_d_max, flatten_grad_d, alpha)
    nc_step_g_new = []
    nc_step_d_new = []
    tmp_d = 0
    for i in range(len(grad_d)):
        nc_step_d_new.append(nc_step_d[tmp_d:size_each_d[i]+tmp_d].view(size_d[i]))
        tmp_d += size_each_d[i]
    tmp_g = 0
    for i in range(len(grad_g)):
        nc_step_g_new.append(nc_step_g[tmp_g:size_each_g[i]+tmp_g].view(size_g[i]))
        tmp_g += size_each_g[i]
    
    grad = grad_d+grad_g
    param = d_param+g_param

    reg = 0.5*sum(torch.sum(torch.square(g)) for g in grad)

    Jgrad = torch.autograd.grad(reg, param)
    
    Jgrad_d = Jgrad[0:len(grad_d)]
    Jgrad_g = Jgrad[len(grad_d):]
    del Jgrad
    
    final_grad_d = [g + nc + gamma*j for g, nc, j in zip(grad_d, nc_step_d_new, Jgrad_d)]
    final_grad_g = [g + nc + gamma*j for g, nc, j in zip(grad_g, nc_step_g_new, Jgrad_g)]

    # backward
    for p,g in zip(disc.parameters(), final_grad_d):
        p.grad = g.detach()
    for p,g in zip(gen.parameters(), final_grad_g):
        p.grad = g.detach()

    opt_disc.step()
    opt_gen.step()

    return opt_disc, opt_gen, nc_step_g, nc_step_d, e_d_max, e_g_min, flatten_grad_d, flatten_grad_g

### Helper function
# For Negative Curvature Step

def nc_step_for_g(v, e, grad, lr):
    sign = torch.sign(v*grad)
    d = lr* torch.abs(e) * v * sign
    d_zero = torch.zeros_like(d)

    # If the minimum eigenvalue is < 0: then we do the update; otherwise just update with a zero vector
    # assign_ops = torch.where(e < 0, lambda: _apply_step(d, _vars), lambda: _apply_step(d_zero, _vars))
    if e < 0:
        assign_ops = d.to(device)
    else:
        assign_ops = d_zero.to(device)
    return assign_ops

def nc_step_for_d(v, e, grad, lr):
    sign = torch.sign(v*grad)
    d = lr* torch.abs(e) * v * sign
    d_zero = torch.zeros_like(d)

    # If the  maximum eigenvalue is > 0: then we do the update; otherwise just update with a zero vector
    # assign_ops = torch.where(e < 0, lambda: _apply_step(d, _vars), lambda: _apply_step(d_zero, _vars))
    if e > 0:
        assign_ops = d.to(device)
    else:
        assign_ops = d_zero.to(device)
    return assign_ops

def Hessian_vec_prod(flatten_grad, param, vecs):
    vecs = _list_to_tensor(vecs).to(device)
    prod = torch.mean(_list_to_tensor([i.mul(j) for i, j in zip(flatten_grad, vecs)]))
    # prod = _list_to_tensor([g.mul(v) for g, v in zip(flatten_grad, vecs)]).mean()
    #prod = _list_to_tensor([torch.sum(g*v) for g,v in zip(flatten_grad,vecs)]).mean()
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
        v.append(eps*v[i] - v_new)
        v[i+1] = _normalize(v[i+1])

        # Get corresponding eigenvalue
        eigval = torch.sum(torch.multiply(v[i], v_new))
        eigvals.append(eigval)
    
    eigvalue = eigvals[iterations-1]
    eigvector = v[iterations-1]
    
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
            n_each.append(params[i].shape[0]*params[i].shape[1])
    return n, n_each


def _tensor_to_list(tensor):        # one-dim list
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
            tensor_list+=tmp
    return tensor_list

def _list_to_tensor(_list):         # n-dim list --> 1-dim tensor
    _tensor = torch.cat([torch.reshape(layer, [-1]) for layer in _list], axis=0)
    return _tensor.to(device)