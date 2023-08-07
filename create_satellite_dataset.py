import rpcm
import glob
import os
import numpy as np
import srtm4
import shutil
import sys
import json
import rasterio
import os
import argparse

from sat_utils import get_file_id

def rio_open(*args,**kwargs):
    import rasterio
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return rasterio.open(*args,**kwargs)

def get_image_lonlat_aoi(rpc, h, w):
    z = srtm4.srtm4(rpc.lon_offset, rpc.lat_offset)
    cols, rows, alts = [0,w,w,0], [0,0,h,h], [z]*4
    lons, lats = rpc.localization(cols, rows, alts)
    lonlat_coords = np.vstack((lons, lats)).T
    geojson_polygon = {"coordinates": [lonlat_coords.tolist()], "type": "Polygon"}
    x_c = lons.min() + (lons.max() - lons.min())/2
    y_c = lats.min() + (lats.max() - lats.min())/2
    geojson_polygon["center"] = [x_c, y_c]
    return geojson_polygon

#Input: cropped images (tif files only)
def run_ba(img_dir, output_dir):

    from bundle_adjust.cam_utils import SatelliteImage
    from bundle_adjust.ba_pipeline import BundleAdjustmentPipeline
    from bundle_adjust import loader

    # load input data
    os.makedirs(output_dir, exist_ok=True)
    myimages = sorted(glob.glob(img_dir + "/*.tif"))
    myrpcs = [rpcm.rpc_from_geotiff(p) for p in myimages]
    input_images = [SatelliteImage(fn, rpc) for fn, rpc in zip(myimages, myrpcs)]
    ba_input_data = {}
    ba_input_data['in_dir'] = img_dir
    ba_input_data['out_dir'] = os.path.join(output_dir, "ba_files")
    ba_input_data['images'] = input_images
    #print('Input data set!\n')

    # redirect all prints to a bundle adjustment logfile inside the output directory
    os.makedirs(ba_input_data['out_dir'], exist_ok=True)
    path_to_log_file = "{}/bundle_adjust.log".format(ba_input_data['out_dir'])
    print("Running bundle adjustment for RPC model refinement ...")
    print("Path to log file: {}".format(path_to_log_file))
    log_file = open(path_to_log_file, "w+")
    sys.stdout = log_file
    sys.stderr = log_file
    # run bundle adjustment
    #tracks_config = {'FT_reset': True, 'FT_sift_detection': 's2p', 'FT_sift_matching': 'epipolar_based', "FT_K": 300}
    tracks_config = {'FT_reset': False, 'FT_save': True, 'FT_sift_detection': 's2p', 'FT_sift_matching': 'epipolar_based'}
    ba_extra = {"cam_model": "rpc"}
    ba_pipeline = BundleAdjustmentPipeline(ba_input_data, tracks_config=tracks_config, extra_ba_config=ba_extra)
    ba_pipeline.run()
    # close logfile
    sys.stderr = sys.__stderr__
    sys.stdout = sys.__stdout__
    log_file.close()
    print("... done !")
    print("Path to output files: {}".format(ba_input_data['out_dir']))

    # save all bundle adjustment parameters in a temporary directory
    ba_params_dir = os.path.join(ba_pipeline.out_dir, "ba_params")
    os.makedirs(ba_params_dir, exist_ok=True)
    np.save(os.path.join(ba_params_dir, "pts_ind.npy"), ba_pipeline.ba_params.pts_ind)
    np.save(os.path.join(ba_params_dir, "cam_ind.npy"), ba_pipeline.ba_params.cam_ind)
    np.save(os.path.join(ba_params_dir, "pts3d.npy"), ba_pipeline.ba_params.pts3d_ba - ba_pipeline.global_transform)
    np.save(os.path.join(ba_params_dir, "pts2d.npy"), ba_pipeline.ba_params.pts2d)
    fnames_in_use = [ba_pipeline.images[idx].geotiff_path for idx in ba_pipeline.ba_params.cam_prev_indices]
    loader.save_list_of_paths(os.path.join(ba_params_dir, "geotiff_paths.txt"), fnames_in_use)

#Input：sunangles file，ba results，images，dsms
#Output：json file （"img"，  "height"，  "width"，  "sun_elevation"，  "sun_azimuth"，  "acquisition_date"，  "geojson"，  "min_alt"，  "max_alt"，  "rpc"）
def create_dataset_from_DFC2019_data(nerf_dir, aoi_id, img_dir, dfc_dir, output_dir, use_ba=False):

    # create a json file of metadata for each input image
    # contains: h, w, rpc, sun elevation, sun azimuth, acquisition date
    #           + geojson polygon with the aoi of the image
    os.makedirs(output_dir, exist_ok=True)
    path_to_dsm = os.path.join(dfc_dir, "Truth/{}_DSM.tif".format(aoi_id))
    if use_ba:
        from bundle_adjust import loader
        geotiff_paths = loader.load_list_of_paths(os.path.join(output_dir, "ba_files/ba_params/geotiff_paths.txt"))
        geotiff_paths = [p.replace("/pan_crops/", "/crops/") for p in geotiff_paths]
        geotiff_paths = [p.replace("PAN.tif", "RGB.tif") for p in geotiff_paths]
        ba_geotiff_basenames = [os.path.basename(x) for x in geotiff_paths]
        ba_kps_pts3d_ind = np.load(os.path.join(output_dir, "ba_files/ba_params/pts_ind.npy"))
        ba_kps_cam_ind = np.load(os.path.join(output_dir, "ba_files/ba_params/cam_ind.npy"))
        ba_kps_pts2d = np.load(os.path.join(output_dir, "ba_files/ba_params/pts2d.npy"))
    else:
        geotiff_paths = sorted(glob.glob(img_dir + "/*.tif"))

    json_dir = nerf_dir + '/root_dir/'
    os.makedirs(json_dir, exist_ok=True)
    json_dir += '/crops_rpcs_ba_v2/'
    os.makedirs(json_dir, exist_ok=True)
    json_dir += '/'+aoi_id+'/'
    os.makedirs(json_dir, exist_ok=True)

    msi_p = dfc_dir + '/'+aoi_id+'_sunangles.txt'
    sun_angles = np.loadtxt(msi_p, dtype='str') 

    for rgb_p in geotiff_paths:
        d = {}
        d["img"] = os.path.basename(rgb_p)

        src = rio_open(rgb_p)
        d["height"] = int(src.meta["height"])
        d["width"] = int(src.meta["width"])
        original_rpc = rpcm.RPCModel(src.tags(ns='RPC'), dict_format="geotiff")

        for t, sun_angle in enumerate(sun_angles):
            if sun_angle[0] == d["img"]:
                d["sun_elevation"] = sun_angle[1].astype(float)
                d["sun_azimuth"] = sun_angle[2].astype(float)
                d["acquisition_date"] = 'none'
                d["geojson"] = get_image_lonlat_aoi(original_rpc, d["height"], d["width"])         
        src = rio_open(path_to_dsm)
        dsm = src.read()[0, :, :]
        d["min_alt"] = int(np.round(dsm.min() - 1))
        d["max_alt"] = int(np.round(dsm.max() + 1))

        if use_ba:
            # use corrected rpc
            rpc_path = os.path.join(output_dir, "ba_files/rpcs_adj/{}.rpc_adj".format(get_file_id(rgb_p)))
            d["rpc"] = rpcm.rpc_from_rpc_file(rpc_path).__dict__
            #d_out["rpc"] = rpc_rpcm_to_geotiff_format(rpc.__dict__)

            # additional fields for depth supervision
            ba_kps_pts3d_path = os.path.join(output_dir, "ba_files/ba_params/pts3d.npy")
            shutil.copyfile(ba_kps_pts3d_path, os.path.join(json_dir, "pts3d.npy"))
            cam_idx = ba_geotiff_basenames.index(d["img"])
            d["keypoints"] = {"2d_coordinates": ba_kps_pts2d[ba_kps_cam_ind == cam_idx, :].tolist(),
                              "pts3d_indices": ba_kps_pts3d_ind[ba_kps_cam_ind == cam_idx].tolist()}
        else:
            # use original rpc
            d["rpc"] = original_rpc.__dict__

        with open(os.path.join(json_dir, "{}.json".format(get_file_id(rgb_p))), "w") as f:
            json.dump(d, f, indent=2)

    return json_dir

def create_train_test_splits(input_sample_ids, test_percent=0.15, min_test_samples=2):

    def shuffle_array(array):
        import random
        v = array.copy()
        random.shuffle(v)
        return v

    n_samples = len(input_sample_ids)
    input_sample_ids = np.array(input_sample_ids)
    all_indices = shuffle_array(np.arange(n_samples))
    n_test = max(min_test_samples, int(test_percent * n_samples))
    n_train = n_samples - n_test

    train_indices = all_indices[:n_train]
    test_indices = all_indices[-n_test:]

    train_samples = input_sample_ids[train_indices].tolist()
    test_samples = input_sample_ids[test_indices].tolist()

    return train_samples, test_samples

#Input: ../Truth/JAX_068_DSM.txt (there contains origins, size and resolutions in UTM)
#Ouput: Bounding box in longitude/latitude
def read_DFC2019_lonlat_aoi(aoi_id, dfc_dir):
    from bundle_adjust import geo_utils
    if aoi_id[:3] == "JAX":
        zonestring = "17" #"17R"
    elif aoi_id[:3] == "Dji":
        zonestring = "38" #"38N"
    else:
        raise ValueError("AOI not valid. Expected JAX_xxx or Dji_xxx but received {}".format(aoi_id))
    roi = np.loadtxt(os.path.join(dfc_dir, "Truth/" + aoi_id + "_DSM.txt"))

    xoff, yoff, xsize, ysize, resolution = roi[0], roi[1], int(roi[2]), int(roi[2]), roi[3]
    ulx, uly, lrx, lry = xoff, yoff + ysize * resolution, xoff + xsize * resolution, yoff
    #bounding box
    xmin, xmax, ymin, ymax = ulx, lrx, uly, lry
    easts = [xmin, xmin, xmax, xmax, xmin]
    norths = [ymin, ymax, ymax, ymin, ymin]
    lons, lats = geo_utils.lonlat_from_utm(easts, norths, zonestring)
    lonlat_bbx = geo_utils.geojson_polygon(np.vstack((lons, lats)).T)
    return lonlat_bbx

#Input: Bounding box in longitude/latitude, images and rpcs before cropping
#Output: images and rpcs after cropping
def crop_geotiff_lonlat_aoi(geotiff_path, output_path, lonlat_aoi):
    with rasterio.open(geotiff_path, 'r') as src:
        profile = src.profile
        tags = src.tags()
    crop, x, y = rpcm.utils.crop_aoi(geotiff_path, lonlat_aoi)
    rpc = rpcm.rpc_from_geotiff(geotiff_path)
    rpc.row_offset -= y
    rpc.col_offset -= x
    not_pan = len(crop.shape) > 2
    if not_pan:
        profile["height"] = crop.shape[1]
        profile["width"] = crop.shape[2]
    else:
        profile["height"] = crop.shape[0]
        profile["width"] = crop.shape[1]
        profile["count"] = 1
    with rasterio.open(output_path, 'w', **profile) as dst:
        if not_pan:
            dst.write(crop)
        else:
            dst.write(crop, 1)
        dst.update_tags(**tags)
        dst.update_tags(ns='RPC', **rpc.to_geotiff_dict())


def create_satellite_dataset(aoi_id, dfc_dir, output_dir, mode='train', ba=True, crop_aoi=True, splits=False):
    if crop_aoi:
        # prepare crops
        aoi_lonlat = read_DFC2019_lonlat_aoi(aoi_id, dfc_dir)
        img_dir = os.path.join(dfc_dir, "RGB/{}".format(aoi_id))
        #myimages = sorted(glob.glob(img_dir + "/*.tif"))
        with open(os.path.join(args.dfc_dir + '/train.txt'), "r") as f:
            train_imgs = f.read().split("\n")[:-1]

        subdir = aoi_id + '_' + str(len(train_imgs)) + '_imgs'
        myimages = []
        for img in train_imgs:
             myimages.append(img[:-5]+'.tif')
        if mode == 'test':
            subdir += '_tmp'
            with open(os.path.join(args.dfc_dir + '/test.txt'), "r") as f:
                test_imgs = f.read().split("\n")[:-1]
            for img in test_imgs:
                myimages.append(img[:-5]+'.tif')
        print('image list of {} mode:\n{}'.format(mode, myimages))
        output_dir = output_dir + subdir + '/'
        os.makedirs(output_dir, exist_ok=True)

        crops_dir = os.path.join(output_dir, "dataset"+aoi_id+"/")
        nerf_dir = crops_dir
        os.makedirs(crops_dir, exist_ok=True)
        crops_dir = os.path.join(crops_dir, aoi_id+"/")
        os.makedirs(crops_dir, exist_ok=True)

        truth_dir = os.path.join(crops_dir, "Truth/")
        os.makedirs(truth_dir, exist_ok=True)

        dsm_dir = os.path.join(dfc_dir, "Truth/")
        DSMTxtFile = dsm_dir + aoi_id + '_DSM.txt'
        DSMFile = dsm_dir + aoi_id + '_DSM.tif'
        shutil.copyfile(DSMTxtFile, truth_dir + os.path.basename(DSMTxtFile))
        shutil.copyfile(DSMFile, truth_dir + os.path.basename(DSMFile))

        crops_dir = os.path.join(crops_dir, "RGB-crops/")
        os.makedirs(crops_dir, exist_ok=True)
        crops_dir = os.path.join(crops_dir, aoi_id+"/")
        os.makedirs(crops_dir, exist_ok=True)

        for img in myimages:
            out_crop_path = os.path.join(crops_dir, img)
            geotiff_path = img_dir + '/' + img
            crop_geotiff_lonlat_aoi(geotiff_path, out_crop_path, aoi_lonlat)
        img_dir = crops_dir
    else:
        img_dir = os.path.join(dfc_dir, "RGB/{}".format(aoi_id))
    if ba:
        run_ba(img_dir, output_dir)
    json_dir = create_dataset_from_DFC2019_data(nerf_dir, aoi_id, img_dir, dfc_dir, output_dir, use_ba=ba)

    # create train and test splits
    if splits:
        json_files = [os.path.basename(p) for p in glob.glob(os.path.join(json_dir, "*.json"))]
        train_samples, test_samples = create_train_test_splits(json_files)
        with open(os.path.join(json_dir, "train.txt"), "w+") as f:
            f.write("\n".join(train_samples))
        with open(os.path.join(json_dir, "test.txt"), "w+") as f:
            f.write("\n".join(test_samples))

    print('finished processing {}ing images !\n'.format(mode))

    return output_dir, img_dir, json_dir

def config_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--aoi_id", type=str, default=None,
                        help='None')
    parser.add_argument("--dfc_dir", type=str, default=None,
                        help='None')
    parser.add_argument("--output_dir", type=str, default=None,
                        help='directory to save the output')
    #parser.add_argument('--splits', action='store_true',
    #                    help='None')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = config_parser()
    #run bundle adjustment with only training images for a clean split
    _, img_dir_train, json_dir_train = create_satellite_dataset(args.aoi_id, args.dfc_dir, args.output_dir, 'train')
    #run bundle adjustment with training and testing images validation
    dir_test, img_dir_test, json_dir_test = create_satellite_dataset(args.aoi_id, args.dfc_dir, args.output_dir, 'test')

    with open(os.path.join(args.dfc_dir + '/test.txt'), "r") as f:
        test_imgs = f.read().split("\n")[:-1]
    for test_img in test_imgs:
        shutil.copyfile(img_dir_test + '/' + test_img[:-5] + '.tif', img_dir_train + '/' + test_img[:-5] + '.tif')
        shutil.copyfile(json_dir_test + '/' + test_img, json_dir_train + '/' + test_img)

    shutil.copyfile(args.dfc_dir + '/train.txt', json_dir_train + '/train.txt')
    shutil.copyfile(args.dfc_dir + '/test.txt', json_dir_train + '/test.txt')

    shutil.rmtree(dir_test)

    print('create_satellite_dataset finished !')