import logging
import torch
import cesp
import time
from func import time_convertor

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"


def SimGA(disc, gen, d_loss, g_loss, opt_gen, opt_disc, gamma, alpha, args):
    d_param = [d for d in disc.parameters()]
    g_param = [g for g in gen.parameters()]

    #### Get grad()
    grad_d = torch.autograd.grad(d_loss, disc.parameters(), create_graph=True)
    grad_g = torch.autograd.grad(g_loss, gen.parameters(), create_graph=True)

    ts = time.time()
    if args.is_eigen == 1:
        v_g_min, e_g_min, flatten_grad_g = cesp.get_min_eigvec(grad_g, g_param)
        v_d_max, e_d_max, flatten_grad_d = cesp.get_min_eigvec(grad_d, d_param)
        nc_step_g = cesp.nc_step_for_g(v_g_min, e_g_min, flatten_grad_g, alpha * 10)
        nc_step_d = cesp.nc_step_for_d(v_d_max, e_d_max, flatten_grad_d, alpha)
    else:
        v_g_min = torch.zeros(cesp.get_num_weights(g_param))
        e_g_min = torch.tensor([0])
        v_d_max = torch.zeros(cesp.get_num_weights(g_param))
        e_d_max = torch.tensor([0])
        flatten_grad_d = cesp._list_to_tensor(grad_d).to(device)
        flatten_grad_g = cesp._list_to_tensor(grad_g).to(device)
        nc_step_g = v_g_min
        nc_step_d = v_d_max
    te = time.time()
    logger.info(
        f"Get eigenpair and nc_step Success, time used: {time_convertor.convert_sec(te-ts)}"
    )

    grad = grad_d + grad_g
    param = d_param + g_param

    reg = 0.0 * sum(torch.sum(torch.square(g)) for g in grad)

    ts = time.time()
    Jgrad = torch.autograd.grad(reg, param)
    te = time.time()
    logger.info(
        f"Get Jacobian Gradient Success, time used: {time_convertor.convert_sec(te-ts)}"
    )

    Jgrad_d = Jgrad[0 : len(grad_d)]
    Jgrad_g = Jgrad[len(grad_d) :]
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

    return (
        opt_disc,
        opt_gen,
        nc_step_g,
        nc_step_d,
        e_d_max,
        e_g_min,
        flatten_grad_d,
        flatten_grad_g,
    )


def ConOpt(disc, gen, d_loss, g_loss, opt_gen, opt_disc, gamma, alpha, args):
    d_param = [d for d in disc.parameters()]
    g_param = [g for g in gen.parameters()]

    #### Get grad()
    grad_d = torch.autograd.grad(
        d_loss, disc.parameters(), retain_graph=True, create_graph=True
    )
    grad_g = torch.autograd.grad(
        g_loss, gen.parameters(), retain_graph=True, create_graph=True
    )

    ts = time.time()
    if args.is_eigen == 1:
        v_g_min, e_g_min, flatten_grad_g = cesp.get_min_eigvec(grad_g, g_param)
        v_d_max, e_d_max, flatten_grad_d = cesp.get_min_eigvec(grad_d, d_param)
        nc_step_g = cesp.nc_step_for_g(v_g_min, e_g_min, flatten_grad_g, alpha * 10)
        nc_step_d = cesp.nc_step_for_d(v_d_max, e_d_max, flatten_grad_d, alpha)
    else:
        v_g_min = torch.zeros(cesp.get_num_weights(g_param))
        e_g_min = torch.tensor([0])
        v_d_max = torch.zeros(cesp.get_num_weights(g_param))
        e_d_max = torch.tensor([0])
        flatten_grad_d = cesp._list_to_tensor(grad_d).to(device)
        flatten_grad_g = cesp._list_to_tensor(grad_g).to(device)
        nc_step_g = v_g_min
        nc_step_d = v_d_max
    te = time.time()
    logger.info(
        f"Get eigenpair and nc_step Success, time used: {time_convertor.convert_sec(te-ts)}"
    )

    grad = grad_d + grad_g
    param = d_param + g_param

    # reg_d = 0.5*sum(torch.sum(g**2) for g in grad_d)
    # reg_g = 0.5*sum(torch.sum(g**2) for g in grad_g)
    # reg = 0.5*sum(torch.sum(g**2) for g in grad)
    reg = 0.5 * sum(torch.sum(torch.square(g)) for g in grad)

    # Jgrad_d = torch.autograd.grad(reg_d, d_param)
    # Jgrad_g = torch.autograd.grad(reg_g, g_param)
    ts = time.time()
    Jgrad = torch.autograd.grad(reg, param)
    te = time.time()
    logger.info(
        f"Get Jacobian Gradient Success, time used: {time_convertor.convert_sec(te-ts)}"
    )

    Jgrad_d = Jgrad[0 : len(grad_d)]
    Jgrad_g = Jgrad[len(grad_d) :]
    del Jgrad

    final_grad_d = [g + gamma * j for j, g in zip(Jgrad_d, grad_d)]
    final_grad_g = [g + gamma * j for j, g in zip(Jgrad_g, grad_g)]

    for p, g in zip(disc.parameters(), final_grad_d):
        p.grad = g.detach()

    for p, g in zip(gen.parameters(), final_grad_g):
        p.grad = g.detach()

    opt_disc.step()
    opt_gen.step()

    return (
        opt_disc,
        opt_gen,
        nc_step_g,
        nc_step_d,
        e_d_max,
        e_g_min,
        flatten_grad_d,
        flatten_grad_g,
    )


def CESP(disc, gen, d_loss, g_loss, opt_gen, opt_disc, gamma, alpha, args):
    d_param = [d for d in disc.parameters()]
    g_param = [g for g in gen.parameters()]

    size_d, size_each_d = cesp.get_num_weights_each_param(d_param)
    size_g, size_each_g = cesp.get_num_weights_each_param(g_param)

    #### Get grad()
    grad_d = torch.autograd.grad(
        d_loss, disc.parameters(), retain_graph=True, create_graph=True
    )
    grad_g = torch.autograd.grad(
        g_loss, gen.parameters(), retain_graph=True, create_graph=True
    )

    ts = time.time()
    v_g_min, e_g_min, flatten_grad_g = cesp.get_min_eigvec(grad_g, g_param)
    v_d_max, e_d_max, flatten_grad_d = cesp.get_min_eigvec(grad_d, d_param)
    te = time.time()
    logger.info(f"Get eigenpair, time used: {time_convertor.convert_sec(te-ts)}")

    ts = time.time()
    nc_step_g = cesp.nc_step_for_g(v_g_min, e_g_min, flatten_grad_g, alpha * 10)
    nc_step_d = cesp.nc_step_for_d(v_d_max, e_d_max, flatten_grad_d, alpha)
    te = time.time()
    logger.info(
        f"Get eigenpair and nc_step Success, time used: {time_convertor.convert_sec(te-ts)}"
    )

    nc_step_g_new = []
    nc_step_d_new = []
    tmp_d = 0
    for i in range(len(grad_d)):
        nc_step_d_new.append(nc_step_d[tmp_d : size_each_d[i] + tmp_d].view(size_d[i]))
        tmp_d += size_each_d[i]
    tmp_g = 0
    for i in range(len(grad_g)):
        nc_step_g_new.append(nc_step_g[tmp_g : size_each_g[i] + tmp_g].view(size_g[i]))
        tmp_g += size_each_g[i]

    grad = grad_d + grad_g
    param = d_param + g_param

    reg = 0.0 * sum(torch.sum(torch.square(g)) for g in grad)

    Jgrad = torch.autograd.grad(reg, param)

    Jgrad_d = Jgrad[0 : len(grad_d)]
    Jgrad_g = Jgrad[len(grad_d) :]
    del Jgrad

    final_grad_d = [g + nc for g, nc in zip(grad_d, nc_step_d_new)]
    final_grad_g = [g + nc for g, nc in zip(grad_g, nc_step_g_new)]

    # backward
    for p, g in zip(disc.parameters(), final_grad_d):
        p.grad = g.detach()
    for p, g in zip(gen.parameters(), final_grad_g):
        p.grad = g.detach()

    opt_disc.step()
    opt_gen.step()

    return (
        opt_disc,
        opt_gen,
        nc_step_g,
        nc_step_d,
        e_d_max,
        e_g_min,
        flatten_grad_d,
        flatten_grad_g,
    )


def CNCO(disc, gen, d_loss, g_loss, opt_gen, opt_disc, gamma, alpha, args):
    d_param = [d for d in disc.parameters()]
    g_param = [g for g in gen.parameters()]

    size_d, size_each_d = cesp.get_num_weights_each_param(d_param)
    size_g, size_each_g = cesp.get_num_weights_each_param(g_param)

    #### Get grad()
    grad_d = torch.autograd.grad(
        d_loss, disc.parameters(), retain_graph=True, create_graph=True
    )
    grad_g = torch.autograd.grad(
        g_loss, gen.parameters(), retain_graph=True, create_graph=True
    )

    ts = time.time()
    v_g_min, e_g_min, flatten_grad_g = cesp.get_min_eigvec(grad_g, g_param)
    v_d_max, e_d_max, flatten_grad_d = cesp.get_min_eigvec(grad_d, d_param)

    nc_step_g = cesp.nc_step_for_g(v_g_min, e_g_min, flatten_grad_g, alpha * 10)
    nc_step_d = cesp.nc_step_for_d(v_d_max, e_d_max, flatten_grad_d, alpha)
    te = time.time()
    logger.info(
        f"Get eigenpair and nc_step Success, time used: {time_convertor.convert_sec(te-ts)}"
    )
    nc_step_g_new = []
    nc_step_d_new = []
    tmp_d = 0
    for i in range(len(grad_d)):
        nc_step_d_new.append(nc_step_d[tmp_d : size_each_d[i] + tmp_d].view(size_d[i]))
        tmp_d += size_each_d[i]
    tmp_g = 0
    for i in range(len(grad_g)):
        nc_step_g_new.append(nc_step_g[tmp_g : size_each_g[i] + tmp_g].view(size_g[i]))
        tmp_g += size_each_g[i]

    grad = grad_d + grad_g
    param = d_param + g_param

    reg = 0.5 * sum(torch.sum(torch.square(g)) for g in grad)

    ts = time.time()
    Jgrad = torch.autograd.grad(reg, param)
    te = time.time()
    logger.info(
        f"Get Jacobian Gradient Success, time used: {time_convertor.convert_sec(te-ts)}"
    )

    Jgrad_d = Jgrad[0 : len(grad_d)]
    Jgrad_g = Jgrad[len(grad_d) :]
    del Jgrad

    final_grad_d = [
        g + nc + gamma * j for g, nc, j in zip(grad_d, nc_step_d_new, Jgrad_d)
    ]
    final_grad_g = [
        g + nc + gamma * j for g, nc, j in zip(grad_g, nc_step_g_new, Jgrad_g)
    ]

    # backward
    for p, g in zip(disc.parameters(), final_grad_d):
        p.grad = g.detach()
    for p, g in zip(gen.parameters(), final_grad_g):
        p.grad = g.detach()

    opt_disc.step()
    opt_gen.step()

    return (
        opt_disc,
        opt_gen,
        nc_step_g,
        nc_step_d,
        e_d_max,
        e_g_min,
        flatten_grad_d,
        flatten_grad_g,
    )
