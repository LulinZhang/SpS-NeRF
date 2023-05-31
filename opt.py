"""
This script defines the input parameters that can be customized from the command line
"""

import argparse
import datetime
import json
import os

def Test_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_id", type=str, default=None,
                        help='exp_name when training SpS-NeRF')
    #parser.add_argument("--aoi_id", type=str, default=None,
    #                    help='None')
    parser.add_argument("--logs_dir", type=str, default=None,
                        help='logs_dir when training SpS-NeRF')
    parser.add_argument("--output_dir", type=str, default=None,
                        help='directory to save the output')
    parser.add_argument("--epoch_number", type=int, default=28,
                        help='epoch_number when training SpS-NeRF')
    parser.add_argument("--split", type=str, default='val',
                        help='None')

    return parser.parse_args()

def printArgs(args):
    print('--aoi_id: ', args.aoi_id)
    print('--beta: ', args.beta)
    print('--sc_lambda: ', args.sc_lambda)
    print('--mapping: ', args.mapping)
    print('--inputdds: ', args.inputdds)
    print('--ds_lambda: ', args.ds_lambda)
    print('--ds_drop: ', args.ds_drop)
    print('--GNLL: ', args.GNLL)
    print('--usealldepth: ', args.usealldepth)
    print('--guidedsample: ', args.guidedsample)
    print('--margin: ', args.margin)
    print('--stdscale: ', args.stdscale)
    print('--corrscale: ', args.corrscale)
    print('--model: ', args.model)
    print('--exp_name: ', args.exp_name)
    print('--lr: ', args.lr)
    print('--n_samples: ', args.n_samples)
    print('--n_importance: ', args.n_importance)
    print('------------------------------')
    print('--root_dir: ', args.root_dir)
    print('--img_dir: ', args.img_dir)
    print('--ckpts_dir: ', args.ckpts_dir)
    print('--logs_dir: ', args.logs_dir)
    print('--gt_dir: ', args.gt_dir)
    print('--cache_dir: ', args.cache_dir)
    print('--ckpt_path: ', args.ckpt_path)
    print('--gpu_id: ', args.gpu_id)
    print('--batch_size: ', args.batch_size)
    print('--img_downscale: ', args.img_downscale)
    print('--max_train_steps: ', args.max_train_steps)
    print('--save_every_n_epochs: ', args.save_every_n_epochs)
    print('--fc_units: ', args.fc_units)
    print('--fc_layers: ', args.fc_layers)
    print('--noise_std: ', args.noise_std)
    print('--chunk: ', args.chunk)
    print('--ds_noweights: ', args.ds_noweights)
    print('--first_beta_epoch: ', args.first_beta_epoch)
    print('--t_embbeding_tau: ', args.t_embbeding_tau)
    print('--t_embbeding_vocab: ', args.t_embbeding_vocab)

def Train_parser():
    parser = argparse.ArgumentParser()

    # input paths
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of the input dataset')
    parser.add_argument('--img_dir', type=str, default=None,
                        help='Directory where the images are located (if different than root_dir)')
    parser.add_argument("--ckpts_dir", type=str, default="ckpts",
                        help="output directory to save trained models")
    parser.add_argument("--logs_dir", type=str, default="logs",
                        help="output directory to save experiment logs")
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='directory where the ground truth DSM is located (if available)')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='directory where cache for the current dataset is found')
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="pretrained checkpoint path to load")

    # other basic stuff and dataset options
    parser.add_argument("--exp_name", type=str, default=None,
                        help="experiment name")
    parser.add_argument("--model", type=str, default='sps-nerf', choices=['nerf', 's-nerf', 'sat-nerf', 'sps-nerf'],
                        help="which NeRF to use")
    parser.add_argument("--gpu_id", type=int, required=True,
                        help="GPU that will be used")

    # training and network configuration
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size (number of input rays per iteration)')
    parser.add_argument('--img_downscale', type=float, default=1.0,
                        help='downscale factor for the input images')
    parser.add_argument('--max_train_steps', type=int, default=300000,
                        help='number of training iterations')
    parser.add_argument('--save_every_n_epochs', type=int, default=4,
                        help="save checkpoints and debug files every n epochs")
    parser.add_argument('--fc_units', type=int, default=512,
                        help='number of fully connected units in the main block of layers')
    parser.add_argument('--fc_layers', type=int, default=8,
                        help='number of fully connected layers in the main block of layers')
    parser.add_argument('--n_samples', type=int, default=64,
                        help='number of coarse scale discrete points per input ray')
    parser.add_argument('--n_importance', type=int, default=0,
                        help='number of fine scale discrete points per input ray')
    parser.add_argument('--noise_std', type=float, default=0.0,
                        help='standard deviation of noise added to sigma to regularize')
    parser.add_argument('--chunk', type=int, default=1024*5,
                        help='maximum number of rays that can be processed at once without memory issues')

    # other sat-nerf specific stuff
    parser.add_argument('--sc_lambda', type=float, default=0.,
                        help='float that multiplies the solar correction auxiliary loss')
    parser.add_argument('--ds_lambda', type=float, default=0.,
                        help='float that multiplies the depth supervision auxiliary loss')
    parser.add_argument('--ds_drop', type=float, default=0.25,
                        help='portion of training steps at which the depth supervision loss will be dropped')
    parser.add_argument('--ds_noweights', action='store_true',
                        help='do not use reprojection errors to weight depth supervision loss')
    parser.add_argument('--first_beta_epoch', type=int, default=2,
                        help='portion of training steps at which the depth supervision loss will be dropped')
    parser.add_argument('--t_embbeding_tau', type=int, default=4,
                        help='portion of training steps at which the depth supervision loss will be dropped')
    parser.add_argument('--t_embbeding_vocab', type=int, default=30,
                        help='portion of training steps at which the depth supervision loss will be dropped')

    #SpS-NeRF add-on
    parser.add_argument('--aoi_id', type=str, default="JAX_068",
                        help='aoi_id')
    parser.add_argument('--inputdds', type=str, default="DenseDepth",
                        help='the folder to the dense depth files')
    parser.add_argument('--beta', action='store_true',  #Recommendation for SpS-NeRF: NOT present in the command-line argument
                        help='by default, do not use beta for transient uncertainty')
    parser.add_argument('--mapping', action='store_true',    #Recommendation for SpS-NeRF: present in the command-line argument
                        help='by default, do not use positional encoding')   
    parser.add_argument('--GNLL', action='store_true',    #Recommendation for SpS-NeRF: NOT present in the command-line argument
                        help='by default, use MSE depth loss instead of Gaussian negative log likelihood loss')    
    parser.add_argument('--usealldepth', action='store_true',    #Recommendation for SpS-NeRF: NOT present in the command-line argument
                        help='by default, use only a subset of depth which meets the condition of R_sub in equation 6 in SpS-NeRF article')    
    parser.add_argument('--guidedsample', action='store_true',    #Recommendation for SpS-NeRF: present in the command-line argument
                        help='by default, do not apply depth-guided sampling')    
    parser.add_argument('--margin', type=float, default=0.0001,
                        help='so that the pts with correlation scores equal to 1 has the std value of margin, instead of 0. (m in equation 5 in SpS-NeRF article)')
    parser.add_argument('--stdscale', type=float, default=1,
                        help='so that the pts with correlation scores close to 0 has the std value of stdscale, instead of 1. (gama in equation 5 in SpS-NeRF article)')
    parser.add_argument('--corrscale', type=float, default=1,
                        help='scale the correlation for dense depth from different resolution (1 for ZM=4, 0.7 for ZM=8)')   #not used


    args = parser.parse_args()

    if (args.model != 'sps-nerf'):
        args.GNLL = False
        args.usealldepth = True
        args.guidedsample = False

    exp_id = args.config_name if args.exp_name is None else args.exp_name
    args.exp_name = exp_id
    print("\nRunning {} - Using gpu {}\n".format(args.exp_name, args.gpu_id))

    os.makedirs("{}/{}".format(args.logs_dir, args.exp_name), exist_ok=True)
    with open("{}/{}/opts.json".format(args.logs_dir, args.exp_name), "w") as f:
        json.dump(vars(args), f, indent=2)


    return args

