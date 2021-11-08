import torch
import torch.nn as nn
import numpy as np
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def d_loss_function(real_labels, d_out_real, d_out_fake):
    d_loss_real = -torch.mean(real_labels * torch.log(d_out_real))                                  
    d_loss_fake = -torch.mean(real_labels * torch.log(1-d_out_fake))
    d_loss = d_loss_fake+d_loss_real
    return d_loss

def g_loss_function(real_labels, d_out_fake):
    g_loss = -torch.mean(real_labels * torch.log(d_out_fake))  
    return g_loss

def SimGA(disc, gen, d_loss, g_loss, opt_gen, opt_disc, real_labels, d_out_real, d_out_fake):
    
    #### Get grad()
    grad_d = torch.autograd.grad(d_loss, disc.parameters())
    grad_g = torch.autograd.grad(g_loss, gen.parameters())

    len_g = len(grad_g)
    len_d = len(grad_d)

    d_param = [d for d in disc.parameters()]                                     
    g_param = [g for g in gen.parameters()]                                                                     

    # start = time.time()

    e_g_min = torch.Tensor([1000]).to(device)
    e_d_max = torch.Tensor([0]).to(device)

    for i in range(len_d+len_g):
        index = 0
        if i < len_d:
            hessian = torch.autograd.functional.hessian(d_loss_function, (real_labels, d_out_real, d_out_fake), create_graph=True)
 
            index+=1
            for j in hessian:
                e_d, v_d = torch.eig(j[0].reshape(j[0].size(0), j[0].size(2)), eigenvectors=True)
                if max(e_d[:,0]) > e_d_max:
                    e_d_max = max(e_d[:,0])
                    # for i in range(len(e_d[:,0])):
                    #     if e_d[i,0] == e_d_max:
                    #         v_d_max = v_d[i]
            
        else:
            hessian = torch.autograd.functional.hessian(g_loss_function, (real_labels, d_out_fake), create_graph=True)
            index+=1
            for j in hessian:

                e_g, v_g = torch.eig(j[0].reshape(j[0].size(0), j[0].size(2)), eigenvectors=True)
                
                if min(e_g[:,0]) < e_g_min:
                    e_g_min = min(e_g[:,0])
                    # for i in range(len(e_g[:,0])):
                    #     if e_g[i,0] == e_g_min:
                    #         v_g_min = v_g[i]

    # end = time.time()
    # print('subtime', end-start)

    # final_grad_d = [g for g in grad_d]                                     
    # final_grad_g = [g for g in grad_g]                                     

    for p, g in zip(d_param, grad_d):                                     
        p.grad = g                                     

    for p, g in zip(g_param, grad_g):                                     
        p.grad = g                                     

    opt_gen.step()                                     
    opt_disc.step()                                     

    return opt_disc, opt_gen, grad_d, grad_g, e_d_max, e_g_min

def ConOpt(disc, gen, d_loss, g_loss, opt_gen, opt_disc, gamma, real_labels, d_out_real, d_out_fake):
    #### Get grad()
    grad_d = torch.autograd.grad(d_loss, disc.parameters(), create_graph=True)
    grad_g = torch.autograd.grad(g_loss, gen.parameters(), create_graph=True)                            

    len_g = len(grad_g)
    len_d = len(grad_d)

    d_param = [d for d in disc.parameters()]                                     
    g_param = [g for g in gen.parameters()]                                     

    start = time.time()

    e_g_min = torch.Tensor([1000]).to(device)
    e_d_max = torch.Tensor([0]).to(device)

    for i in range(len_d+len_g):
        index = 0
        if i < len_d:
            hessian = torch.autograd.functional.hessian(d_loss_function, (real_labels, d_out_real, d_out_fake), create_graph=True)
 
            index+=1
            for j in hessian:
                e_d, v_d = torch.eig(j[0].reshape(j[0].size(0), j[0].size(2)), eigenvectors=True)
                if max(e_d[:,0]) > e_d_max:
                    e_d_max = max(e_d[:,0])
                    # for i in range(len(e_d[:,0])):
                    #     if e_d[i,0] == e_d_max:
                    #         v_d_max = v_d[i]
            
        else:
            hessian = torch.autograd.functional.hessian(g_loss_function, (real_labels, d_out_fake), create_graph=True)
            index+=1
            for j in hessian:

                e_g, v_g = torch.eig(j[0].reshape(j[0].size(0), j[0].size(2)), eigenvectors=True)
                
                if min(e_g[:,0]) < e_g_min:
                    e_g_min = min(e_g[:,0])
                    # for i in range(len(e_g[:,0])):
                    #     if e_g[i,0] == e_g_min:
                    #         v_g_min = v_g[i]

    end = time.time()
    print('subtime', end - start)

    grad = grad_d + grad_g
    param = d_param + g_param

    reg = 0.5*sum(torch.sum(g**2) for g in grad) 

    Jgrad = torch.autograd.grad(reg, param).to(device)

    final_grad = [g + gamma * j for j, g in zip(Jgrad, grad)]                                    

    for p, g in zip(param, final_grad):                                     
        p.grad = g                                                      

    opt_gen.step()                                     
    opt_disc.step()                                     

    return opt_disc, opt_gen, grad_d, grad_g, e_d_max, e_g_min

def CESP(disc, gen, d_loss, g_loss, opt_gen, opt_disc, alpha, real_labels, d_out_real, d_out_fake):                                   
    #### Get grad()
    grad_d = torch.autograd.grad(d_loss, disc.parameters(), create_graph=True)
    grad_g = torch.autograd.grad(g_loss, gen.parameters(), create_graph=True)
    
    len_g = len(grad_g)
    len_d = len(grad_d)

    d_param = [d for d in disc.parameters()]                                     
    g_param = [g for g in gen.parameters()]

    grad = grad_d + grad_g							# grad
    param = d_param + g_param						# variable
    
    start = time.time()
   
    e_g_min = torch.Tensor([1000]).to(device)
    e_d_max = torch.Tensor([0]).to(device)

    for i in range(len_d+len_g):
        index = 0
        if i < len_d:
            hessian = torch.autograd.functional.hessian(d_loss_function, (real_labels, d_out_real, d_out_fake), create_graph=True)
 
            index+=1
            for j in hessian:
                e_d, v_d = torch.eig(j[0].reshape(j[0].size(0), j[0].size(2)), eigenvectors=True)
                if max(e_d[:,0]) > e_d_max:
                    e_d_max = max(e_d[:,0])
                    for i in range(len(e_d[:,0])):
                        if e_d[i,0] == e_d_max:
                            v_d_max = v_d[i]
            
        else:
            hessian = torch.autograd.functional.hessian(g_loss_function, (real_labels, d_out_fake), create_graph=True)
            index+=1
            for j in hessian:

                e_g, v_g = torch.eig(j[0].reshape(j[0].size(0), j[0].size(2)), eigenvectors=True)
                
                if min(e_g[:,0]) < e_g_min:
                    e_g_min = min(e_g[:,0])
                    for i in range(len(e_g[:,0])):
                        if e_g[i,0] == e_g_min:
                            v_g_min = v_g[i]
    
    

    # v_g_min, e_g_min = get_min_eigvec(grad_g, g_param, 'generator')
    # v_d_max, e_d_max = get_min_eigvec(grad_d, d_param, 'discriminator')
    
    end = time.time()
    print('time', end-start)
    
    nc_step_g = nc_step_for_g(v_g_min, e_g_min, g_param, alpha)
    nc_step_d = nc_step_for_d(v_d_max, e_d_max, d_param, alpha)    

    final_grad_d = [g + nc for g, nc in zip(grad_d, nc_step_d)]
    final_grad_g = [g + nc for g, nc in zip(grad_g, nc_step_g)]

    for p,g in zip(d_param, final_grad_d):
        p.grad = g
    for p,g in zip(g_param, final_grad_g):
        p.grad = g

    opt_gen.step()
    opt_disc.step()

    return opt_disc, opt_gen, grad_d, grad_g, nc_step_g, nc_step_d, e_d_max, e_g_min

def CNCO(disc, gen, d_loss, g_loss, opt_gen, opt_disc, gamma, alpha, real_labels, d_out_real, d_out_fake):
    #### Get grad()
    grad_d = torch.autograd.grad(d_loss, disc.parameters(), create_graph=True)
    grad_g = torch.autograd.grad(g_loss, gen.parameters(), create_graph=True)                                     

    len_g = len(grad_g)
    len_d = len(grad_d)

    d_param = [d for d in disc.parameters()]                                     
    g_param = [g for g in gen.parameters()]

    e_g_min = torch.Tensor([1000]).to(device)
    e_d_max = torch.Tensor([0]).to(device)
    for i in range(len_d+len_g):
        if i < len_d:
            hessian = torch.autograd.functional.hessian(d_loss_function, (real_labels, d_out_real, d_out_fake), create_graph=True)
            for j in hessian:

                e_d, v_d = torch.eig(j[0].reshape(j[0].size(0), j[0].size(2)), eigenvectors=True)
                
                if max(e_d[:,0]) > e_d_max:
                    e_d_max = max(e_d[:,0])
                    for i in range(len(e_d[:,0])):
                        if e_d[i,0] == e_d_max:
                            v_d_max = v_d[i]

        else:
            hessian = torch.autograd.functional.hessian(g_loss_function, (real_labels, d_out_fake), create_graph=True)
            # e_g, v_g = torch.eig(hessian, eigenvectors=True)
            # if min(e_g[:,0]) < e_g_min:
            #     e_g_min = min(e_g[:,0])
            #     for i in range(len(e_g)):
            #         if e_g[i,0] == e_g_min:
            #             v_g_min = v_g[i]
            for j in hessian:

                e_g, v_g = torch.eig(j[0].reshape(j[0].size(0), j[0].size(2)), eigenvectors=True)
                
                if min(e_g[:,0]) < e_g_min:
                    e_g_min = min(e_g[:,0])
                    for i in range(len(e_g[:,0])):
                        if e_g[i,0] == e_g_min:
                            v_g_min = v_g[i]
    
    # v_g_min, e_g_min = get_min_eigvec(g_loss, g_param, gen)
    # v_d_max, e_d_max = get_min_eigvec(d_loss, d_param, disc)

    nc_step_g = nc_step_for_g(v_g_min, e_g_min, g_param, alpha)
    nc_step_d = nc_step_for_d(v_d_max, e_d_max, d_param, alpha)

    reg_d = 0.5*sum(torch.sum(g**2) for g in grad_d)                                     
    reg_g = 0.5*sum(torch.sum(g**2) for g in grad_g)                                     

    Jgrad_d = torch.autograd.grad(reg_d, d_param)
    Jgrad_g = torch.autograd.grad(reg_g, g_param)

    final_grad_d = [g + nc + gamma*j for g, nc, j in zip(grad_d, nc_step_d, Jgrad_d)]
    final_grad_g = [g + nc + gamma*j for g, nc, j in zip(grad_g, nc_step_g, Jgrad_g)]

    for p,g in zip(d_param, final_grad_d):
        p.grad = g
    for p,g in zip(g_param, final_grad_g):
        p.grad = g

    opt_gen.step()
    opt_disc.step()

    return opt_disc, opt_gen, grad_d, grad_g, nc_step_g, nc_step_d, e_d_max, e_g_min

### Helper function
# For Negative Curvature Step

def nc_step_for_g(v, e, _vars, lr):
    d = lr* torch.abs(e) * v
    d_zero = torch.zeros_like(d)
    
    # If the minimum eigenvalue is < 0: then we do the update; otherwise just update with a zero vector
    # assign_ops = torch.where(e < 0, lambda: _apply_step(d, _vars), lambda: _apply_step(d_zero, _vars))
    if e < 0:
        assign_ops = d
    else:
        assign_ops = d_zero
    return assign_ops

def nc_step_for_d(v, e, _vars, lr):
    d = lr* torch.abs(e) * v
    d_zero = torch.zeros_like(d)
    
    # If the  maximum eigenvalue is > 0: then we do the update; otherwise just update with a zero vector
    # assign_ops = torch.where(e < 0, lambda: _apply_step(d, _vars), lambda: _apply_step(d_zero, _vars))
    if e > 0:
        assign_ops = d
    else:
        assign_ops = d_zero
    return assign_ops

def Hessian_vec_prod(flatten_grad, param, vecs):
    vecs = _list_to_tensor(vecs)

    
    # start = time.time()
    # prod = sum([(g*v).sum() for g,v in zip(flatten_grad,vecs)])
    # end = time.time()
    # print('subsubtime1', end-start)
    start = time.time()
    prod = torch.sum(_list_to_tensor([torch.sum(g*v) for g,v in zip(flatten_grad,vecs)]))
    end = time.time()
    print('subsubtime2', end-start)
    # start = time.time()
    # prod = torch.sum(_list_to_tensor([torch.sum(torch.multiply(g, torch.t(v))) for g, v in zip(flatten_grad, vecs)])).to(device)
    # end = time.time()
    # print('subsubtime3', end-start)
    prod.backward(retain_graph=True)
    return [p.grad.detach() for p in param]

def get_min_eigvec(grad, param, model_str_name):
    iterations = 10
    eps = 3
    v = [_get_initial_vector(param)]

    eigvals = []
    
    # flatten_grad = []
    # for i in range(len(grad)):
    #     flatten_grad+=_tensor_to_list(grad[i])

    # flatten_grad = []
    # for i in range(len(grad)):
    #     if len(grad[i].shape) == 1:
    #         for j  in range(grad[i].shape[0]):
    #             flatten_grad.append(grad[i][j])
    #     else:
    #         for j in range(grad[i].shape[0]):
    #             for k in range(grad[i].shape[1]):
    #                 flatten_grad.append(grad[i][j][k])
    
    # flatten_grad = torch.Tensor().to(device)
    # for i in range(len(grad)):
    #     if len(grad[i].shape) != 1:
    #         flatten_grad = torch.cat([flatten_grad, torch.flatten(grad[i])])
    #     else:
    #         flatten_grad = torch.cat([flatten_grad, grad[i]])
    
    
    # flatten_grad = torch.Tensor().to(device)
    # for i in range(len(grad)):
    #     flatten_grad_i = torch.flatten(grad[i])
    #     flatten_grad = torch.cat([flatten_grad, flatten_grad_i]).to(device)
    
    size = get_num_weights_each_param(param)
    size_ = get_num_weights(param)
    flatten_grad = torch.zeros(size_).to(device)
    
    tmp = 0
    for i in range(len(grad)):
        flatten_grad[tmp:size[i]+tmp] = torch.flatten(grad[i])
        tmp += size[i]
    
    for i in range(iterations):
        # Power iteration with the shifted Hessian
        start = time.time()
        hvp = Hessian_vec_prod(flatten_grad, param, v)
        end = time.time()
        print('subtime', end - start)
        v_new = _list_to_tensor(hvp)
        v.append(eps*v[i] - v_new)
        v[i+1] = _normalize(v[i+1])

        # Get corresponding eigenvalue
        eigval = torch.sum(torch.multiply(v[i], v_new))
        eigvals.append(eigval)
    
    # print(eigvals)
    # print(v[1].shape)
    # idx = iterations -1 #tf.cast(tf.argmin(eigvals[3:iterations-1]), tf.int32)
    # e = torch.gather(eigvals, idx)
    # v = torch.gather(v, idx)
    
    # start = time.time()
    if model_str_name == 'discriminator':
        eigvalue = max(eigvals)
        for i in range(len(eigvals)):
            if eigvalue == eigvals[i]:
                eigvector = v[i]
    elif model_str_name == 'generator':
        eigvalue = min(eigvals)
        for i in range(len(eigvals)):
            if eigvalue == eigvals[i]:
                eigvector = v[i]
    # end = time.time()
    # print('subtime', end - start)
    
    # _sign = torch.sign(torch.multiply(torch.stack(tuple(flatten_grad)), eigvector))
    _sign = torch.sign(torch.multiply(flatten_grad, eigvector))
    
    eigvector *= _sign

    return eigvector, eigvalue

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
    shape = [list(layer.shape) for layer in params]
    n = []
    for sh in shape:
        n_layer = 1
        for dim in sh:
            n_layer*=dim
        n.append(n_layer)
    return n


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

def _list_to_tensor(_list):
    _tensor = torch.cat([torch.reshape(layer, [-1]) for layer in _list], axis=0)
    return _tensor.to(device)