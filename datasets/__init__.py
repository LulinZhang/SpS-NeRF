from .satellite import SatelliteDataset
from .satellite_depth import SatelliteDataset_depth
from .satellite_rgb_dep import SatelliteRGBDEPDataset

def load_dataset(args, split):

    outputs = []
    if args.model == 'sps-nerf':        #for sps-nerf, load rgb and depth at the same time, depth is padded to the same size as rgb
        d1 = SatelliteRGBDEPDataset(root_dir=args.root_dir,
                     img_dir=args.img_dir if args.img_dir is not None else args.root_dir,
                     split=split,
                     cache_dir=args.cache_dir,
                     gt_dir=args.gt_dir+"/",
                     aoi_id=args.aoi_id,
                     img_downscale=args.img_downscale, inputdds=args.inputdds, corrscale=args.corrscale, stdscale=args.stdscale, margin=args.margin)
    else:           #for nerf and sat-nerf, load rgb
        d1 = SatelliteDataset(root_dir=args.root_dir,
                     img_dir=args.img_dir if args.img_dir is not None else args.root_dir,
                     split=split,
                     cache_dir=args.cache_dir,
                     img_downscale=args.img_downscale)
    outputs.append(d1)
    
    if args.model == 'sat-nerf':         #for sat-nerf, load sparse depth individually, which is not the same size as rgb
        d2 = SatelliteDataset_depth(root_dir=args.root_dir,
                     img_dir=args.img_dir if args.img_dir is not None else args.root_dir,
                     split=split,
                     cache_dir=args.cache_dir,
                     img_downscale=args.img_downscale)
        outputs.append(d2)

    return outputs

