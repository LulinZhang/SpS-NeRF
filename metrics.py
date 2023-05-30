"""
This script defines the evaluation metrics and the loss functions
"""

import torch
import numpy as np
from kornia.losses import ssim as ssim_

class NerfLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss_dict = {}
        loss_dict['coarse_color'] = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss_dict['fine_color'] = self.loss(inputs['rgb_fine'], targets)
        loss = sum(l for l in loss_dict.values())
        return loss, loss_dict

def uncertainty_aware_loss(loss_dict, inputs, gt_rgb, typ, beta_min=0.05):
    beta = torch.sum(inputs[f'weights_{typ}'].unsqueeze(-1) * inputs['beta_coarse'], -2) + beta_min
    loss_dict[f'{typ}_color'] = ((inputs[f'rgb_{typ}'] - gt_rgb) ** 2 / (2 * beta ** 2)).mean()
    loss_dict[f'{typ}_logbeta'] = (3 + torch.log(beta).mean()) / 2  # +3 to make c_b positive since beta_min = 0.05
    return loss_dict

def solar_correction(loss_dict, inputs, typ, lambda_sc=0.05):
    # computes the solar correction terms defined in Shadow NeRF and adds them to the dictionary of losses
    sun_sc = inputs[f'sun_sc_{typ}'].squeeze()
    term2 = torch.sum(torch.square(inputs[f'transparency_sc_{typ}'].detach() - sun_sc), -1)
    term3 = 1 - torch.sum(inputs[f'weights_sc_{typ}'].detach() * sun_sc, -1)
    loss_dict[f'{typ}_sc_term2'] = lambda_sc/3. * torch.mean(term2)
    loss_dict[f'{typ}_sc_term3'] = lambda_sc/3. * torch.mean(term3)
    return loss_dict

class SNerfLoss(torch.nn.Module):
    def __init__(self, lambda_sc=0.05):
        super().__init__()
        self.lambda_sc = lambda_sc
        self.loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss_dict = {}
        typ = 'coarse'
        loss_dict[f'{typ}_color'] = self.loss(inputs[f'rgb_{typ}'], targets)
        if self.lambda_sc > 0:
            loss_dict = solar_correction(loss_dict, inputs, typ, self.lambda_sc)
        if 'rgb_fine' in inputs:
            typ = 'fine'
            loss_dict[f'{typ}_color'] = self.loss(inputs[f'rgb_{typ}'], targets)
            if self.lambda_sc > 0:
                loss_dict = solar_correction(loss_dict, inputs, typ, self.lambda_sc)
        loss = sum(l for l in loss_dict.values())
        return loss, loss_dict

class SatNerfLoss(torch.nn.Module):
    def __init__(self, lambda_sc=0.0):
        super().__init__()
        self.lambda_sc = lambda_sc

    def forward(self, inputs, targets):
        loss_dict = {}
        typ = 'coarse'
        loss_dict = uncertainty_aware_loss(loss_dict, inputs, targets, typ)
        if self.lambda_sc > 0:
            loss_dict = solar_correction(loss_dict, inputs, typ, self.lambda_sc)
        if 'rgb_fine' in inputs:
            typ = 'fine'
            loss_dict = uncertainty_aware_loss(loss_dict, inputs, targets, typ)
            if self.lambda_sc > 0:
                loss_dict = solar_correction(loss_dict, inputs, typ, self.lambda_sc)
        loss = sum(l for l in loss_dict.values())
        return loss, loss_dict

class DepthLoss(torch.nn.Module):
    def __init__(self, lambda_ds=1.0, GNLL=False, usealldepth=True, margin=0, stdscale=1):
        super().__init__()
        self.lambda_ds = lambda_ds/3.
        self.GNLL = GNLL
        self.usealldepth = usealldepth
        self.margin=margin
        self.stdscale=stdscale
        if self.GNLL == True:
            self.loss = torch.nn.GaussianNLLLoss()
        else:
            self.loss = torch.nn.MSELoss(reduce=False)

    def forward(self, inputs, targets, weights=1., target_valid_depth=None, target_std=None):

        def is_not_in_expected_distribution(depth_mean, depth_var, depth_measurement_mean, depth_measurement_std):
            delta_greater_than_expected = ((depth_mean - depth_measurement_mean).abs() - depth_measurement_std) > 0.
            var_greater_than_expected = depth_measurement_std.pow(2) < depth_var
            return torch.logical_or(delta_greater_than_expected, var_greater_than_expected)

        def ComputeSubsetDepthLoss(inputs, typ, target_mean, target_weight, target_valid_depth, target_std):
            if target_valid_depth == None:
                print('target_valid_depth is None! Use all the target_depth by default! target_mean.shape[0]', target_mean.shape[0])
                target_valid_depth = torch.ones(target_mean.shape[0])
            z_vals = inputs[f'z_vals_{typ}'][np.where(target_valid_depth.cpu()>0)]
            pred_mean = inputs[f'depth_{typ}'][np.where(target_valid_depth.cpu()>0)]

            pred_weight = inputs[f'weights_{typ}'][np.where(target_valid_depth.cpu()>0)]
            if pred_mean.shape[0] == 0:
                print('ZERO target_valid_depth in this depth loss computation! target_weight.device: ', target_weight.device)
                return torch.zeros((1,), device=target_weight.device, requires_grad=True)

            pred_var = ((z_vals - pred_mean.unsqueeze(-1)).pow(2) * pred_weight).sum(-1) + 1e-5
            target_weight = target_weight[np.where(target_valid_depth.cpu()>0)]
            target_mean = target_mean[np.where(target_valid_depth.cpu()>0)]
            target_std = target_std[np.where(target_valid_depth.cpu()>0)]
            #target_std = self.stdscale*(torch.ones_like(target_weight) - target_weight) + torch.ones_like(target_weight)*self.margin

            apply_depth_loss = torch.ones(target_mean.shape[0])
            if self.usealldepth == False:
                apply_depth_loss = is_not_in_expected_distribution(pred_mean, pred_var, target_mean, target_std)

            pred_mean = pred_mean[apply_depth_loss]
            if pred_mean.shape[0] == 0:
                print('ZERO apply_depth_loss in this depth loss computation!')
                return torch.zeros((1,), device=target_weight.device, requires_grad=True)

            pred_var = pred_var[apply_depth_loss]
            target_mean = target_mean[apply_depth_loss]

            numerator = float(pred_mean.shape[0])
            denominator = float(target_valid_depth.shape[0])

            if self.GNLL == True:   
                loss = numerator/denominator*self.loss(pred_mean, target_mean, pred_var)
                return loss
            else:
                loss = numerator/denominator*target_weight[apply_depth_loss]*self.loss(pred_mean, target_mean)
                return loss


        loss_dict = {}
        typ = 'coarse'
        if self.usealldepth == False:
            loss_dict[f'{typ}_ds'] = ComputeSubsetDepthLoss(inputs, typ, targets, weights, target_valid_depth, target_std)
        else:
            loss_dict[f'{typ}_ds'] = self.loss(inputs['depth_coarse'], targets)

        if 'depth_fine' in inputs:
            typ = 'fine'
            if self.usealldepth == False:
                loss_dict[f'{typ}_ds'] = ComputeSubsetDepthLoss(inputs, typ, targets, weights, target_valid_depth, target_std)
            else:
                loss_dict[f'{typ}_ds'] = self.loss(inputs['depth_fine'], targets)

        if self.usealldepth == False:
            # no need to apply weights here because it is already done in function ComputeSubsetDepthLoss
            for k in loss_dict.keys():
                loss_dict[k] = self.lambda_ds * torch.mean(loss_dict[k])
        else:
            # apply weights
            for k in loss_dict.keys():
                loss_dict[k] = self.lambda_ds * torch.mean(weights * loss_dict[k])

        loss = sum(l for l in loss_dict.values())
        return loss, loss_dict

def load_loss(args):
    if args.model == "nerf":
        loss_function = NerfLoss()
    elif args.model == "s-nerf":
        loss_function = SNerfLoss(lambda_sc=args.sc_lambda)
    elif args.model == "sat-nerf" or args.model == "sps-nerf":
        if args.beta == True:
            loss_function = SatNerfLoss(lambda_sc=args.sc_lambda)
        else:
            loss_function = SNerfLoss(lambda_sc=args.sc_lambda)      
    else:
        raise ValueError(f'model {args.model} is not valid')
    return loss_function

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

def ssim(image_pred, image_gt):
    """
    image_pred and image_gt: (1, 3, H, W)
    important: kornia==0.5.3
    """
    return torch.mean(ssim_(image_pred, image_gt, 3))
