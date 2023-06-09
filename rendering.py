"""
This script renders the input rays that are used to feed the NeRF model
It discretizes each ray in the input batch into a set of 3d points at different depths of the scene
Then the nerf model takes these 3d points (and the ray direction, optionally, as in the original nerf)
and predicts a volume density at each location (sigma) and the color with which it appears
"""

import torch
import math
import numpy as np

def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Args:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Returns:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples

def sample_3sigma(low_3sigma, high_3sigma, N, det, near, far, device=None, gt=False):
    t_vals = torch.linspace(0., 1., steps=N, device=device)
    step_size = (high_3sigma - low_3sigma) / (N - 1)
    bin_edges = (low_3sigma.unsqueeze(-1) * (1.-t_vals) + high_3sigma.unsqueeze(-1) * (t_vals)).clamp(near, far)
    factor = (bin_edges[..., 1:] - bin_edges[..., :-1]) / step_size.unsqueeze(-1)
    x_in_3sigma = torch.linspace(-3., 3., steps=(N - 1), device=device)
    bin_weights = factor * (1. / math.sqrt(2 * np.pi) * torch.exp(-0.5 * x_in_3sigma.pow(2))).unsqueeze(0).expand(*bin_edges.shape[:-1], N - 1)
    return sample_pdf(bin_edges, bin_weights, N, det=det)

def compute_samples_around_depth(res, N_samples, z_vals, perturb, near, far, device=None):
    pred_depth = res['depth']
    pred_weight = res['weights']
    sampling_std = (((z_vals - pred_depth.unsqueeze(-1)).pow(2) * pred_weight).sum(-1)).sqrt()

    depth_min = pred_depth - 3. * sampling_std
    depth_max = pred_depth + 3. * sampling_std

    z_vals_2 = sample_3sigma(depth_min, depth_max, N_samples, perturb == 0., near, far, device=device)

    return z_vals_2

def GenerateGuidedSamples(res, z_vals, N_samples, perturb, near, far, mode='test', valid_depth=None, target_depths=None, target_std=None, device=None, margin=0, stdscale=1):
    z_vals_2 = torch.empty_like(z_vals)

    # sample around the predicted depth from the first half of samples
    z_vals_2 = compute_samples_around_depth(res, N_samples, z_vals, perturb, near[0, 0], far[0, 0], device=device)

    if mode == 'train':
        assert valid_depth != None, 'valid_depth missing in training batch!'
        target_depth = torch.flatten(target_depths[:, 0][np.where(valid_depth.cpu()>0)])
        
        target_weight = target_depths[:, 1][np.where(valid_depth.cpu()>0)]
        #target_std = stdscale*(torch.ones_like(target_weight) - target_weight) + torch.ones_like(target_weight)*margin
        target_std = torch.flatten(target_std[np.where(valid_depth.cpu()>0)])

        depth_min = target_depth - 3. * target_std
        depth_max = target_depth + 3. * target_std

        z_vals_2_bkp = z_vals_2.clone()
        # sample with in 3 sigma of the GT depth
        gt_samples = sample_3sigma(depth_min, depth_max, N_samples, perturb == 0., near[0, 0], far[0, 0], device=device,gt=True)
        z_vals_2[np.where(valid_depth.cpu()>0)] = gt_samples

    return z_vals_2

def render_rays(models, args, rays, ts, mode='test', valid_depth=None, target_depths=None, target_std=None):
    # get config values
    N_samples = args.n_samples
    N_importance = args.n_importance
    variant = args.model
    use_disp = False
    perturb = 1.0

    # get rays
    rays_o, rays_d, near, far = rays[:, 0:3],  rays[:, 3:6], rays[:, 6:7], rays[:, 7:8]

    # sample depths for coarse model
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)
    if not use_disp:  # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else:  # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    if perturb > 0:  # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    # discretize rays into a set of 3d points (N_rays, N_samples_, 3), one point for each depth of each ray
    xyz_coarse = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)

    # run coarse model
    typ = "coarse"
    if variant == "s-nerf":
        from models.snerf import inference
        sun_d = rays[:, 8:11]
        # render using main set of rays
        result = inference(models[typ], args, xyz_coarse, z_vals, rays_d=None, sun_d=sun_d)
        if args.sc_lambda > 0:
            # solar correction
            xyz_coarse = rays_o.unsqueeze(1) + sun_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)
            result_ = inference(models[typ], args, xyz_coarse, z_vals, rays_d=None, sun_d=sun_d)
            result['weights_sc'] = result_["weights"]
            result['transparency_sc'] = result_["transparency"]
            result['sun_sc'] = result_["sun"]
    elif variant == "sat-nerf" or variant == "sps-nerf":
        from models.satnerf import inference
        sun_d = rays[:, 8:11]
        rays_t = None 
        if args.beta == True:
            rays_t = models['t'](ts) if ts is not None else None
        result = inference(models[typ], args, xyz_coarse, z_vals, rays_d=None, sun_d=sun_d, rays_t=rays_t)
        if(args.guidedsample == True and variant == 'sps-nerf'):       #guidedsample is only for sps-nerf
            z_vals_2 = GenerateGuidedSamples(result, z_vals, N_samples, perturb, near, far, mode=mode, valid_depth=valid_depth, target_depths=target_depths, target_std=target_std, device=rays.device, margin=args.margin, stdscale=args.stdscale).detach()
            z_vals_2, _ = torch.sort(z_vals_2, -1)
            z_vals_unsort = torch.cat([z_vals, z_vals_2], -1)
            z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_2], -1), -1)
            xyz_coarse = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples*2, 3)
            result = inference(models[typ], args, xyz_coarse, z_vals, rays_d=None, sun_d=sun_d, rays_t=rays_t, z_vals_unsort=z_vals_unsort)        
        if args.sc_lambda > 0:
            # solar correction
            xyz_coarse = rays_o.unsqueeze(1) + sun_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)
            result_tmp = inference(models[typ], args, xyz_coarse, z_vals, rays_d=None, sun_d=sun_d, rays_t=rays_t)
            result['weights_sc'] = result_tmp["weights"]
            result['transparency_sc'] = result_tmp["transparency"]
            result['sun_sc'] = result_tmp["sun"]
    else:
        # classic nerf
        from models.nerf import inference
        result = inference(models[typ], args, xyz_coarse, z_vals, rays_d=rays_d)
    result_ = {}
    for k in result.keys():
        result_[f"{k}_{typ}"] = result[k]

    # run fine model
    if N_importance > 0:

        # sample depths for fine model
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, result_['weights_coarse'][:, 1:-1],
                             N_importance, det=(perturb == 0)).detach()
        # detach so that grad doesn't propogate to weights_coarse from here
        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        # discretize rays for fine model
        xyz_fine = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples+N_importance, 3)

        typ = "fine"
        if variant == "s-nerf":
            sun_d = rays[:, 8:11]
            # render using main set of rays
            result = inference(models[typ], args, xyz_fine, z_vals, rays_d=rays_d_, sun_d=sun_d)
            if args.sc_lambda > 0:
                # solar correction
                xyz_fine = rays_o.unsqueeze(1) + sun_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)
                result_ = inference(models[typ], args, xyz_fine, z_vals, rays_d=None, sun_d=sun_d, rays_t=None)
                result['weights_sc'] = result_["weights"]
                result['transparency_sc'] = result_["transparency"]
                result['sun_sc'] = result_["sun"]
        elif variant == "sat-nerf" or variant == "sps-nerf":
            sun_d = rays[:, 8:11]
            rays_t = None  
            if args.beta == True:
                rays_t = models['t'](ts) if ts is not None else None
            result = inference(models[typ], args, xyz_fine, z_vals, rays_d=None, sun_d=sun_d, rays_t=rays_t)
            if args.sc_lambda > 0:
                # solar correction
                xyz_fine = rays_o.unsqueeze(1) + sun_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)
                result_ = inference(models[typ], args, xyz_fine, z_vals, rays_d=None, sun_d=sun_d, rays_t=rays_t)
                result['weights_sc'] = result_["weights"]
                result['transparency_sc'] = result_["transparency"]
                result['sun_sc'] = result_["sun"]           
        else:
            result = inference(models[typ], args, xyz_fine, z_vals, rays_d=rays_d)
        for k in result.keys():
            result_["{}_{}".format(k, typ)] = result[k]

    return result_
