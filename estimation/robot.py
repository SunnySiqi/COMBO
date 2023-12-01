import torch.optim
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.optim.sgd import SGD
import model.model as module_arch

# adapted from: https://github.com/pipilurj/ROBOT/tree/main

class MetaSGD(SGD):
    def __init__(self, net, *args, **kwargs):
        super(MetaSGD, self).__init__(*args, **kwargs)
        self.net = net

    def set_parameter(self, current_module, name, parameters):
        if '.' in name:
            name_split = name.split('.')
            module_name = name_split[0]
            rest_name = '.'.join(name_split[1:])
            for children_name, children in current_module.named_children():
                if module_name == children_name:
                    self.set_parameter(children, rest_name, parameters)
                    break
        else:
            current_module._parameters[name] = parameters

    def meta_step(self, old_params, grads, if_update = False):
        group = self.param_groups[0]
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        lr = group['lr']
        params = []
        for name_p, grad in zip(old_params, grads):
            # parameter.detach_()
            name, parameter = name_p
            if weight_decay != 0:
                grad_wd = grad.add(parameter, alpha=weight_decay)
            else:
                grad_wd = grad
            if momentum != 0 and 'momentum_buffer' in self.state[parameter]:
                buffer = self.state[parameter]['momentum_buffer']
                grad_b = buffer.mul(momentum).add(grad_wd, alpha=1-dampening)
            else:
                grad_b = grad_wd
            if nesterov:
                grad_n = grad_wd.add(grad_b, alpha=momentum)
            else:
                grad_n = grad_b
            if if_update:
                self.set_parameter(self.net, name, parameter.add(grad_n, alpha=-lr))
            else:
                # print("what is lr", lr)
                if grad_n is not None:
                    params.append(parameter - lr*grad_n)
        if not if_update:
            return params

class sig_t(nn.Module):
    def __init__(self, device, num_classes, init=2):
        super(sig_t, self).__init__()

        self.register_parameter(name='w', param=nn.parameter.Parameter(-init*torch.ones(num_classes, num_classes)))

        self.w.to(device)

        co = torch.ones(num_classes, num_classes)
        ind = np.diag_indices(co.shape[0])
        co[ind[0], ind[1]] = torch.zeros(co.shape[0])
        self.co = co.to(device)
        self.identity = torch.eye(num_classes).to(device)


    def forward(self):
        sig = torch.sigmoid(self.w)
        T = self.identity.detach() + sig*self.co.detach()
        T = F.normalize(T, p=1, dim=1)
        return T
def reweighting_revision_loss( out, target, T):
    loss = 0.
    eps = 1e-10
    out_softmax = F.softmax(out, dim=1)
    for i in range(len(target)):
        temp_softmax = out_softmax[i]
        temp = out[i]
        temp = torch.unsqueeze(temp, 0)
        temp_softmax = torch.unsqueeze(temp_softmax, 0)
        temp_target = target[i]
        temp_target = torch.unsqueeze(temp_target, 0)
        pro1 = temp_softmax[:, target[i]].detach()
        out_T = torch.matmul(T.t(), temp_softmax.t().detach())
        out_T = out_T.t()
        pro2 = out_T[:, target[i]]
        beta = pro1 / (pro2 + eps)
        cross_loss = F.cross_entropy(temp, temp_target)
        _loss = beta * cross_loss
        loss += _loss
    return loss / len(target)

def get_correct_percentage(T):
    diagonal = torch.diagonal(T, 0)
    column_sums = torch.sum(T, dim=0)
    fraction_correct = diagonal/column_sums
    return fraction_correct


def foward_loss( out, target, T):
    out_softmax = F.softmax(out, dim=1)
    p_T = torch.matmul(out_softmax , T)
    cross_loss = F.nll_loss(torch.log(p_T), target)
    return cross_loss, out_softmax, p_T


def DMI_loss(output, target):
    outputs = F.softmax(output, dim=1)
    targets = target.reshape(target.size(0), 1).cpu()
    y_onehot = torch.FloatTensor(target.size(0), 10).zero_()
    y_onehot.scatter_(1, targets, 1)
    y_onehot = y_onehot.transpose(0, 1).cuda()
    mat = y_onehot @ outputs
    return -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)

def robot_update_t(trans, eval_loader, device, model, meta_optimizer, cls_num, loss_name, outer_obj):
    
    # if  type(model) == type(module_arch.resnet34(cls_num)):
    #     pseudo_net = module_arch.resnet34(num_classes=cls_num)
    # else:
    #     pseudo_net = module_arch.resnet50(num_classes=cls_num)
    # print("PSEUDO NET", type(pseudo_net))
    # pseudo_net.load_state_dict(model.state_dict())
    # pseudo_net = pseudo_net.to(device)
    # pseudo_net.train()
    # old_params = [(n, p) for (n,p) in pseudo_net.named_parameters() if p.requires_grad]
    # trans_matrix = trans()
    # trans_matrix = trans_matrix.detach().cpu().numpy()
    # print("ROBOT TRANSITION MATRIX BEFORE!!!!", trans_matrix)
    for batch_idx, (inputs, labels, indexs, _) in enumerate(eval_loader):
        # inputs, labels = inputs.to(device), labels.to(device)
        # #get output
        # _, pseudo_outputs = pseudo_net(inputs)
        # # trans_matrix = torch.autograd.Variable(trans_matrix, requires_grad=True)
        # if loss_name == "reweight":
        #     pseudo_loss = reweighting_revision_loss(pseudo_outputs, labels, trans_matrix)
        # elif loss_name == "forward":
        #     pseudo_loss, out_softmax, p_T = foward_loss(pseudo_outputs, labels, trans_matrix)
        # # gather proxy inputs
        # pseudo_grads = torch.autograd.grad(pseudo_loss, [p[1] for p in old_params], create_graph=True, allow_unused=True)
        # pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=0.1)
        # pseudo_optimizer.load_state_dict(model_optimizer.state_dict())
        # pseudo_optimizer.meta_step(old_params, pseudo_grads, if_update=True)

        # del pseudo_grads

        meta_inputs, meta_labels = inputs.to(device), labels.to(device)
        _, meta_outputs = model(meta_inputs)
        # prob = F.softmax(meta_outputs, dim=1)
        # prob = prob.t()
        # out_forward = torch.matmul(trans().t(), prob)
        # out_forward = out_forward.t()
        meta_loss, out_softmax, p_T = foward_loss(meta_outputs, meta_labels, trans())
        # if outer_obj == "rce":
        #     one_hot = torch.zeros(len(meta_labels), cls_num).cuda().scatter_(1, meta_labels.view(-1, 1),  cls_num).cuda()
        #     one_hot = F.softmax(one_hot)
        #     meta_loss_vector = out_forward*(F.log_softmax(meta_outputs, dim=1)-torch.log(one_hot)) - torch.mul(out_forward, F.log_softmax(meta_outputs))
        # elif outer_obj == 'mae':
        #     yhat_meta_1 = F.softmax(meta_outputs, dim=-1)
        #     first_index = to_var(torch.arange(meta_outputs.size(0)).type(
        #         torch.LongTensor), requires_grad=False)
        #     yhat_meta_1 = yhat_meta_1[first_index, meta_labels]
        #     meta_loss_vector = 2*torch.mean(1. - yhat_meta_1)
        # elif outer_obj == 'gce':
        #     loss = 0
        #     for i in range(meta_outputs.size(0)):
        #         loss += (1.0 - (meta_outputs[i][meta_labels[i]]) ** 1) / 1
        #     meta_loss_vector = loss / meta_outputs.size(0)
        # elif outer_obj == 'dmi':
        #     meta_loss_vector = DMI_loss(meta_outputs, meta_labels)
        # else:
        #     raise NotImplementedError(f"{outer_obj} is not implemented")
        # meta_loss = torch.mean(meta_loss_vector)
        meta_optimizer.zero_grad()
        meta_loss.backward(retain_graph=True)
        meta_optimizer.step()
    trans_matrix = trans()
    trans_matrix = trans_matrix.detach().cpu().numpy()
    print("ROBOT TRANSITION MATRIX", trans_matrix)
    return trans_matrix


   

