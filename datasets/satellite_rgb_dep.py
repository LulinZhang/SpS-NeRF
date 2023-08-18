"""
This script defines the dataloader for a dataset of multi-view satellite images
"""

import numpy as np
import os

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

import rasterio
import rpcm
import glob
import sat_utils
from .cal_rmse_depth import cal_rmse_depth


def get_rays(cols, rows, rpc, min_alt, max_alt):
    """
            Draw a set of rays from a satellite image
            Each ray is defined by an origin 3d point + a direction vector
            First the bounds of each ray are found by localizing each pixel at min and max altitude
            Then the corresponding direction vector is found by the difference between such bounds
            Args:
                cols: 1d array with image column coordinates
                rows: 1d array with image row coordinates
                rpc: RPC model with the localization function associated to the satellite image
                min_alt: float, the minimum altitude observed in the image
                max_alt: float, the maximum altitude observed in the image
            Returns:
                rays: (h*w, 8) tensor of floats encoding h*w rays
                      columns 0,1,2 correspond to the rays origin
                      columns 3,4,5 correspond to the direction vector
                      columns 6,7 correspond to the distance of the ray bounds with respect to the camera
            """

    min_alts = float(min_alt) * np.ones(cols.shape)
    max_alts = float(max_alt) * np.ones(cols.shape)

    # assume the points of maximum altitude are those closest to the camera
    lons, lats = rpc.localization(cols, rows, max_alts)
    x_near, y_near, z_near = sat_utils.latlon_to_ecef_custom(lats, lons, max_alts)
    xyz_near = np.vstack([x_near, y_near, z_near]).T

    # similarly, the points of minimum altitude are the furthest away from the camera
    lons, lats = rpc.localization(cols, rows, min_alts)
    x_far, y_far, z_far = sat_utils.latlon_to_ecef_custom(lats, lons, min_alts)
    xyz_far = np.vstack([x_far, y_far, z_far]).T

    # define the rays origin as the nearest point coordinates
    rays_o = xyz_near

    # define the unit direction vector
    d = xyz_far - xyz_near
    rays_d = d / np.linalg.norm(d, axis=1)[:, np.newaxis]

    # assume the nearest points are at distance 0 from the camera
    # the furthest points are at distance Euclidean distance(far - near)
    fars = np.linalg.norm(d, axis=1)
    nears = float(0) * np.ones(fars.shape)

    # create a stack with the rays origin, direction vector and near-far bounds
    rays = torch.from_numpy(np.hstack([rays_o, rays_d, nears[:, np.newaxis], fars[:, np.newaxis]]))
    rays = rays.type(torch.FloatTensor)
    return rays

def load_tensor_from_rgb_geotiff(img_path, downscale_factor, imethod=Image.BILINEAR): #Image.BICUBIC leads to noisy pixel errors
    with rasterio.open(img_path, 'r') as f:
        img = np.transpose(f.read(), (1, 2, 0)) / 255.
    h, w = img.shape[:2]
    if downscale_factor > 1:
        w = int(w // downscale_factor)
        h = int(h // downscale_factor)
        img = np.transpose(img, (2, 0, 1))
        img = T.Resize(size=(h, w), interpolation=imethod)(torch.Tensor(img))
        img = np.transpose(img.numpy(), (1, 2, 0))
    img = T.ToTensor()(img)  # (3, h, w)
    rgbs = img.view(3, -1).permute(1, 0)  # (h*w, 3)
    rgbs = rgbs.type(torch.FloatTensor)
    return rgbs


class SatelliteRGBDEPDataset(Dataset):
    def __init__(self, root_dir, img_dir, split="train", img_downscale=1.0, cache_dir=None, gt_dir=None, aoi_id=None, inputdds="DenseDepth", corrscale=1, stdscale=1, margin=0):
        """
        NeRF Satellite Dataset
        Args:
            root_dir: string, directory containing the json files with all relevant metadata per image
            img_dir: string, directory containing all the satellite images (may be different from root_dir)
            split: string, either 'train' or 'val'
            img_downscale: float, image downscale factor
            cache_dir: string, directory containing precomputed rays
        """
        self.inputdds = inputdds
        self.corrscale = corrscale
        self.stdscale = stdscale
        self.margin = margin
        self.json_dir = root_dir
        self.img_dir = img_dir
        self.cache_dir = cache_dir
        self.gt_dir = gt_dir
        self.aoi_id = aoi_id
        self.train = split == "train"
        self.img_downscale = float(img_downscale)
        self.white_back = False

        print('Load SatelliteRGBDEPDataset with corrscale: ', self.corrscale)

        assert os.path.exists(root_dir), f"root_dir {root_dir} does not exist"
        assert os.path.exists(img_dir), f"img_dir {img_dir} does not exist"

        # load scaling params
        if not os.path.exists(f"{self.json_dir}/scene.loc"):
            self.init_scaling_params()
        else:
            print(f"{self.json_dir}/scene.loc already exist, hence skipped scaling")
        d = sat_utils.read_dict_from_json(os.path.join(self.json_dir, "scene.loc"))
        self.center = torch.tensor([float(d["X_offset"]), float(d["Y_offset"]), float(d["Z_offset"])])
        self.range = torch.max(torch.tensor([float(d["X_scale"]), float(d["Y_scale"]), float(d["Z_scale"])]))

        # load dataset split
        if self.train:
            self.load_train_split()
        else:
            self.load_val_split()

    def load_train_split(self):
        with open(os.path.join(self.json_dir, "train.txt"), "r") as f:
            json_files = f.read().split("\n")
        json_files = json_files[:-1]
        n_train_ims = len(json_files)
        self.json_files = [os.path.join(self.json_dir, json_p) for json_p in json_files]
        self.all_rays, self.all_rgbs, self.all_ids = self.load_data(self.json_files, verbose=True)

        self.depth_dir = self.json_dir+'/'+self.inputdds+'/'
        self.all_deprays, self.all_depths, self.all_valid_depth, self.all_depth_stds = self.load_depth_data(self.json_files, self.depth_dir, verbose=True)

    def load_val_split(self):
        with open(os.path.join(self.json_dir, "test.txt"), "r") as f:
            json_files = f.read().split("\n")
        json_files = json_files[:-1]
        self.json_files = [os.path.join(self.json_dir, json_p) for json_p in json_files]
        # add an extra image from the training set to the validation set (for debugging purposes)
        with open(os.path.join(self.json_dir, "train.txt"), "r") as f:
            json_files = f.read().split("\n")
        json_files = json_files[:-1]
        n_train_ims = len(json_files)
        self.all_ids = [i + n_train_ims for i, j in enumerate(self.json_files)]
        self.json_files = [os.path.join(self.json_dir, json_files[0])] + self.json_files
        self.all_ids = [0] + self.all_ids

    def init_scaling_params(self):
        print("Could not find a scene.loc file in the root directory, creating one...")
        print("Warning: this can take some minutes")
        all_json = glob.glob("{}/*.json".format(self.json_dir))
        all_rays = []
        for json_p in all_json:
            d = sat_utils.read_dict_from_json(json_p)
            h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
            rpc = sat_utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)
            min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
            cols, rows = np.meshgrid(np.arange(w), np.arange(h))
            rays = get_rays(cols.flatten(), rows.flatten(), rpc, min_alt, max_alt)
            all_rays += [rays]
        all_rays = torch.cat(all_rays, 0)
        near_points = all_rays[:, :3]
        far_points = all_rays[:, :3] + all_rays[:, 7:8] * all_rays[:, 3:6]
        all_points = torch.cat([near_points, far_points], 0)

        d = {}
        d["X_scale"], d["X_offset"] = sat_utils.rpc_scaling_params(all_points[:, 0])
        d["Y_scale"], d["Y_offset"] = sat_utils.rpc_scaling_params(all_points[:, 1])
        d["Z_scale"], d["Z_offset"] = sat_utils.rpc_scaling_params(all_points[:, 2])
        sat_utils.write_dict_to_json(d, f"{self.json_dir}/scene.loc")
        print("... done !")

    def load_data(self, json_files, verbose=False):
        """
        Load all relevant information from a set of json files
        Args:
            json_files: list containing the path to the input json files
        Returns:
            all_rays: (N, 11) tensor of floats encoding all ray-related parameters corresponding to N rays
                      columns 0,1,2 correspond to the rays origin
                      columns 3,4,5 correspond to the direction vector
                      columns 6,7 correspond to the distance of the ray bounds with respect to the camera
                      columns 8,9,10 correspond to the sun direction vectors
            all_rgbs: (N, 3) tensor of floats encoding all the rgb colors corresponding to N rays
        """
        all_rgbs, all_rays, all_sun_dirs, all_ids = [], [], [], []
        for t, json_p in enumerate(json_files):

            # read json, image path and id
            if os.path.exists(json_p) == False or os.path.isfile(json_p) == False:
                print(json_p, 'not exist or is not a file, hence skipped')
                continue

            d = sat_utils.read_dict_from_json(json_p)
            img_p = os.path.join(self.img_dir, d["img"])
            img_id = sat_utils.get_file_id(d["img"])

            # get rgb colors
            rgbs = load_tensor_from_rgb_geotiff(img_p, self.img_downscale)

            # get rays
            cache_path = "{}/{}.data".format(self.cache_dir, img_id)
            if self.cache_dir is not None and os.path.exists(cache_path):
                rays = torch.load(cache_path)
            else:
                h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
                rpc = sat_utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)
                min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
                cols, rows = np.meshgrid(np.arange(w), np.arange(h))
                #numpy flatten, default 'C', means in row-major (C-style) order.
                rays = get_rays(cols.flatten(), rows.flatten(), rpc, min_alt, max_alt)
                if self.cache_dir is not None:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    torch.save(rays, cache_path)
            rays = self.normalize_rays(rays)

            # get sun direction
            sun_dirs = self.get_sun_dirs(float(d["sun_elevation"]), float(d["sun_azimuth"]), rays.shape[0])

            all_ids += [t * torch.ones(rays.shape[0], 1)]
            all_rgbs += [rgbs]
            all_rays += [rays]
            all_sun_dirs += [sun_dirs]
            if verbose:
                print("Image {} loaded ( {} / {} )".format(img_id, t + 1, len(json_files)))

        all_ids = torch.cat(all_ids, 0)
        all_rays = torch.cat(all_rays, 0)  # (len(json_files)*h*w, 8)
        all_rgbs = torch.cat(all_rgbs, 0)  # (len(json_files)*h*w, 3)
        all_sun_dirs = torch.cat(all_sun_dirs, 0)  # (len(json_files)*h*w, 3)
        all_rays = torch.hstack([all_rays, all_sun_dirs])  # (len(json_files)*h*w, 11)
        all_rays = all_rays.type(torch.FloatTensor)
        all_rgbs = all_rgbs.type(torch.FloatTensor)
        
        return all_rays, all_rgbs, all_ids

    def scale_depth(self, feature, height, width, depth=1):
        new_height, new_width = int(height/self.img_downscale), int(width/self.img_downscale)
        new_feature = torch.nn.functional.interpolate(feature.reshape(1, 1, height, width, depth), size=(new_height, new_width, depth))     #, mode='bilinear')
        return new_feature.squeeze().reshape(new_height*new_width, depth).squeeze()

    def load_depth_data(self, json_files, depth_dir, verbose=False):
        all_deprays, all_depths, all_sun_dirs, all_weights = [], [], [], []
        all_depth_stds = []
        all_valid_depth = []
        depth_min = 0
        depth_max = 0

        for t, json_p in enumerate(json_files):
            # read json
            d = sat_utils.read_dict_from_json(json_p)
            img_id = sat_utils.get_file_id(d["img"])

            height = d["height"]
            width = d["width"]
            pts2d = []
            idx_cur = 0
            pts2d = np.loadtxt(depth_dir+img_id+"_2DPts.txt", dtype='int')
            pts2d = pts2d.reshape(-1,2)
            pts3d = np.loadtxt(depth_dir+img_id+"_3DPts_ecef.txt", dtype='float')
            pts3d = pts3d.reshape(-1,3)
            current_weights = np.loadtxt(depth_dir+img_id+"_Correl.txt", dtype='float')
            
            valid_depth = torch.zeros(height, width)
            valid_depth[pts2d[:,1], pts2d[:,0]] = torch.ones(pts2d.shape[0])
            valid_depth = valid_depth.flatten()

            CorrelMin = current_weights.min()
            CorrelMax = current_weights.max()
            current_weights = (current_weights-CorrelMin)/(CorrelMax-CorrelMin)
            current_weights = self.corrscale*current_weights

            rpc = sat_utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)
            pts2d = pts2d / self.img_downscale

            # build the sparse batch of rays for depth supervision
            cols, rows = pts2d.T
            min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
            rays = get_rays(cols, rows, rpc, min_alt, max_alt)
            rays = self.normalize_rays(rays)

            # normalize the 3d coordinates of the tie points observed in the current view
            pts3d = torch.from_numpy(pts3d).type(torch.FloatTensor)
            pts3d[:, 0] -= self.center[0]
            pts3d[:, 1] -= self.center[1]
            pts3d[:, 2] -= self.center[2]
            pts3d[:, 0] /= self.range
            pts3d[:, 1] /= self.range
            pts3d[:, 2] /= self.range

            # compute depths
            depths = torch.linalg.norm(pts3d - rays[:, :3], axis=1)
            print('center, range: ', self.center, self.range)
            current_weights = torch.from_numpy(current_weights).type(torch.FloatTensor)
            current_depth_std = self.stdscale*(torch.ones_like(current_weights) - current_weights) + torch.ones_like(current_weights)*self.margin
            cur_depth_min, cur_depth_max = torch.min(depths), torch.max(depths)
            if(t==0):
                depth_min, depth_max = cur_depth_min, cur_depth_max
            else:
                if(cur_depth_min < depth_min):
                    depth_min = cur_depth_min
                if(cur_depth_max > depth_max):
                    depth_max = cur_depth_max
            print("Depth {} loaded ( {} / {} )".format(img_id, t + 1, len(json_files)))
            print('depth range: [{:.5f}, {:.5f}], mean: {:.5f}'.format(torch.min(depths), torch.max(depths), torch.mean(depths)))            
            print('corr  range: [{:.5f}, {:.5f}], mean: {:.5f}'.format(torch.min(current_weights), torch.max(current_weights), torch.mean(current_weights)))            
            print('std   range: [{:.5f}, {:.5f}], mean: {:.5f}'.format(torch.min(current_depth_std), torch.max(current_depth_std), torch.mean(current_depth_std)))

            print('{:.5f} percent of pixels are valid in depth map.'.format(depths.shape[0]*100.0/height/width))

            densedepth_file = depth_dir+img_id+"_3DPts.txt"
            cal_rmse_depth(densedepth_file, self.gt_dir, self.aoi_id)

            depths_padded = torch.zeros(height*width)
            depths_padded[np.where(valid_depth>0)[0]] = depths
            depths_padded = self.scale_depth(depths_padded, height, width)
            
            weights_padded = torch.zeros(height*width)
            weights_padded[np.where(valid_depth>0)[0]] = current_weights
            weights_padded = self.scale_depth(weights_padded, height, width)
            
            depth_std_padded = torch.zeros(height*width)
            depth_std_padded[np.where(valid_depth>0)[0]] = current_depth_std
            depth_std_padded = self.scale_depth(depth_std_padded, height, width)
            
            rays_padded = torch.zeros(height*width, 8)
            rays_padded[np.where(valid_depth>0)[0],:] = rays
            rays_padded = self.scale_depth(rays_padded, height, width, 8)
            
            valid_depth = self.scale_depth(valid_depth, height, width)
            
            all_valid_depth += [valid_depth]
            all_depths += [depths_padded[:, np.newaxis]]
            all_weights += [weights_padded[:, np.newaxis]]
            all_depth_stds += [depth_std_padded]
            all_deprays += [rays_padded]

        all_valid_depth = torch.cat(all_valid_depth, 0)
        all_deprays = torch.cat(all_deprays, 0)  # (len(json_files)*h*w, 8), this should be the same as the one in rgb
        all_depths = torch.cat(all_depths, 0)  # (len(json_files)*h*w, 1)
        print('depth_min, depth_max: ', depth_min, depth_max)
        print('all_depths range: [{:.5f}, {:.5f}], mean: {:.5f}'.format(torch.min(all_depths), torch.max(all_depths), torch.mean(all_depths)))
        all_weights = torch.cat(all_weights, 0)
        all_depth_stds = torch.cat(all_depth_stds, 0)
        print('all_depth_stds range: [{:.5f}, {:.5f}], mean: {:.5f}'.format(torch.min(all_depth_stds), torch.max(all_depth_stds), torch.mean(all_depth_stds)))
        all_depth_stds = all_depth_stds*(depth_max-depth_min)
        print('all_depth_stds range: [{:.5f}, {:.5f}], mean: {:.5f}'.format(torch.min(all_depth_stds), torch.max(all_depth_stds), torch.mean(all_depth_stds)))
        all_depths = torch.hstack([all_depths, all_weights])  # (len(json_files)*h*w, 2)
        all_deprays = all_deprays.type(torch.FloatTensor)
        all_depths = all_depths.type(torch.FloatTensor)


        return all_deprays, all_depths, all_valid_depth, all_depth_stds

    def normalize_rays(self, rays):
        rays[:, 0] -= self.center[0]
        rays[:, 1] -= self.center[1]
        rays[:, 2] -= self.center[2]
        rays[:, 0] /= self.range
        rays[:, 1] /= self.range
        rays[:, 2] /= self.range
        rays[:, 6] /= self.range    #near of the ray
        rays[:, 7] /= self.range    #far of the ray
        return rays

    def get_sun_dirs(self, sun_elevation_deg, sun_azimuth_deg, n_rays):
        """
        Get sun direction vectors
        Args:
            sun_elevation_deg: float, sun elevation in  degrees
            sun_azimuth_deg: float, sun azimuth in degrees
            n_rays: number of rays affected by the same sun direction
        Returns:
            sun_d: (n_rays, 3) 3-valued unit vector encoding the sun direction, repeated n_rays times
        """
        sun_el = np.radians(sun_elevation_deg)
        sun_az = np.radians(sun_azimuth_deg)
        sun_d = np.array([np.sin(sun_az) * np.cos(sun_el), np.cos(sun_az) * np.cos(sun_el), np.sin(sun_el)])
        sun_dirs = torch.from_numpy(np.tile(sun_d, (n_rays, 1)))
        sun_dirs = sun_dirs.type(torch.FloatTensor)
        return sun_dirs

    def get_latlonalt_from_nerf_prediction(self, rays, depth):
        """
        Compute an image of altitudes from a NeRF depth prediction output
        Args:
            rays: (h*w, 11) tensor of input rays
            depth: (h*w, 1) tensor with nerf depth prediction
        Returns:
            lats: numpy vector of length h*w with the latitudes of the predicted points
            lons: numpy vector of length h*w with the longitude of the predicted points
            alts: numpy vector of length h*w with the altitudes of the predicted points
        """

        # convert inputs to double (avoids loss of resolution later when the tensors are converted to numpy)
        rays = rays.double()
        depth = depth.double()

        # use input rays + predicted sigma to construct a point cloud
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
        xyz_n = rays_o + rays_d * depth.view(-1, 1)

        # denormalize prediction to obtain ECEF coordinates
        xyz = xyz_n * self.range
        xyz[:, 0] += self.center[0]
        xyz[:, 1] += self.center[1]
        xyz[:, 2] += self.center[2]

        # convert to lat-lon-alt
        xyz = xyz.data.numpy()
        lats, lons, alts = sat_utils.ecef_to_latlon_custom(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        return lats, lons, alts

    def get_dsm_from_nerf_prediction(self, rays, depth, dsm_path=None, roi_txt=None):
        """
        Compute a DSM from a NeRF depth prediction output
        Args:
            rays: (h*w, 11) tensor of input rays
            depth: (h*w, 1) tensor with nerf depth prediction
            dsm_path (optional): string, path to output DSM, in case you want to write it to disk
            roi_txt (optional): compute the DSM only within the bounds of the region of interest of the txt
        Returns:
            dsm: (h, w) numpy array with the output dsm
        """

        # get point cloud from nerf depth prediction
        lats, lons, alts = self.get_latlonalt_from_nerf_prediction(rays, depth)
        easts, norths = sat_utils.utm_from_latlon(lats, lons)
        cloud = np.vstack([easts, norths, alts]).T

        # (optional) read region of interest, where lidar GT is available
        if roi_txt is not None:
            gt_roi_metadata = np.loadtxt(roi_txt)
            xoff, yoff = gt_roi_metadata[0], gt_roi_metadata[1]
            xsize, ysize = int(gt_roi_metadata[2]), int(gt_roi_metadata[2])
            resolution = gt_roi_metadata[3]
            yoff += ysize * resolution  # weird but seems necessary ?
        else:
            resolution = 0.5
            xmin, xmax = cloud[:, 0].min(), cloud[:, 0].max()
            ymin, ymax = cloud[:, 1].min(), cloud[:, 1].max()
            xoff = np.floor(xmin / resolution) * resolution
            xsize = int(1 + np.floor((xmax - xoff) / resolution))
            yoff = np.ceil(ymax / resolution) * resolution
            ysize = int(1 - np.floor((ymin - yoff) / resolution))

        from plyflatten import plyflatten
        from plyflatten.utils import rasterio_crs, crs_proj
        import utm
        import affine
        import rasterio

        # run plyflatten
        dsm = plyflatten(cloud, xoff, yoff, resolution, xsize, ysize, radius=1, sigma=float("inf"))

        n = utm.latlon_to_zone_number(lats[0], lons[0])
        l = utm.latitude_to_zone_letter(lats[0])
        crs_proj = rasterio_crs(crs_proj("{}{}".format(n, l), crs_type="UTM"))

        # (optional) write dsm to disk
        if dsm_path is not None:
            os.makedirs(os.path.dirname(dsm_path), exist_ok=True)
            profile = {}
            profile["dtype"] = dsm.dtype
            profile["height"] = dsm.shape[0]
            profile["width"] = dsm.shape[1]
            profile["count"] = 1
            profile["driver"] = "GTiff"
            profile["nodata"] = float("nan")
            profile["crs"] = crs_proj
            profile["transform"] = affine.Affine(resolution, 0.0, xoff, 0.0, -resolution, yoff)
            with rasterio.open(dsm_path, "w", **profile) as f:
                f.write(dsm[:, :, 0], 1)

        return dsm

    def __len__(self):
        # compute length of dataset
        if self.train:
            return self.all_rays.shape[0]
        else:
            return len(self.json_files)

    def __getitem__(self, idx):
        # take a batch from the dataset
        if self.train:
            #rays_ref: depth rays padded to the same size of rgb rays, for debug only (to verify the correspondence)
            sample = {"rays": self.all_rays[idx], "rgbs": self.all_rgbs[idx], "ts": self.all_ids[idx].long(), "valid_depth": self.all_valid_depth[idx].long(), "depths": self.all_depths[idx], "rays_ref": self.all_deprays[idx], "depth_std": self.all_depth_stds[idx]}
        else:
            rays, rgbs, _ = self.load_data([self.json_files[idx]])
            ts = self.all_ids[idx] * torch.ones(rays.shape[0], 1)
            d = sat_utils.read_dict_from_json(self.json_files[idx])
            img_id = sat_utils.get_file_id(d["img"])
            h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
            sample = {"rays": rays, "rgbs": rgbs, "ts": ts.long(), "src_id": img_id, "h": h, "w": w}
        return sample
