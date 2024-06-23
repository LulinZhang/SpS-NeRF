import rasterio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from sat_utils import dsm_pointwise_diff
from plyflatten.utils import rasterio_crs, crs_proj
from rasterio.enums import Resampling
from rasterio.transform import Affine
import glob


def ScaleDSM(dsm_in_path, zonestring, dsm_out_path, upscale_factor=1, dst_transform=None, Print=False):
    with rasterio.open(dsm_in_path, 'r') as dataset:
        newHei = int(dataset.height * upscale_factor)
        newWid = int(dataset.width * upscale_factor)
        if Print:
            print('newHei, newWid: ', newHei, newWid)
        data = dataset.read(
            out_shape=(
                dataset.count,
                newHei,
                newWid
            ),
            resampling=Resampling.bilinear
        )
        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (1.0/upscale_factor),
            (1.0/upscale_factor)
        )
        if Print:
            print(transform)

        profile = dataset.profile

        crs_proj1 = rasterio_crs(crs_proj(zonestring, crs_type='UTM'))
        profile.update(transform=transform, height=newHei, width=newWid, crs=crs_proj1)
        
        #if abs(xsize) > 0.0005 or abs(ysize) > 0.0005 or abs(west) > 0.0005 or abs(north) > 0.0005:
        if dst_transform != None:
            profile.update(transform=dst_transform)

        with rasterio.open(dsm_out_path, 'w', **profile) as dst:
            dst.write(data)

def get_dsm_from_dense_depth(densedepth_file, zonestring, dsm_out_path, resolution, roi_txt=None, dst_transform=None):
    pts3d = np.loadtxt(densedepth_file, dtype='float')
    easts = pts3d[:,0]
    norths = pts3d[:,1]
    alts = pts3d[:,2]    
    cloud = np.vstack([easts, norths, alts]).T

    # (optional) read region of interest, where lidar GT is available
    if roi_txt is not None:
        gt_roi_metadata = np.loadtxt(roi_txt)
        xoff, yoff = gt_roi_metadata[0], gt_roi_metadata[1]
        xsize, ysize = int(gt_roi_metadata[2]), int(gt_roi_metadata[2])
        resolution = gt_roi_metadata[3]
        yoff += ysize * resolution  # weird but seems necessary ?
    else:
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

    crs_proj = rasterio_crs(crs_proj(zonestring, crs_type="UTM"))

    # (optional) write dsm to disk
    if dsm_out_path is not None:
        os.makedirs(os.path.dirname(dsm_out_path), exist_ok=True)
        profile = {}
        profile["dtype"] = dsm.dtype
        profile["height"] = dsm.shape[0]
        profile["width"] = dsm.shape[1]
        profile["count"] = 1
        profile["driver"] = "GTiff"
        profile["nodata"] = float("nan")
        profile["crs"] = crs_proj
        if dst_transform != None:
            profile["transform"] = dst_transform
        else:
            profile["transform"] = affine.Affine(resolution, 0.0, xoff, 0.0, -resolution, yoff)
        with rasterio.open(dsm_out_path, "w", **profile) as f:
            f.write(dsm[:, :, 0], 1)


def get_info(aoi_id):
    if aoi_id[0:3] == 'JAX':
        zonestring = "17R"
        resolution = 0.35
        upscale_factor = 0.35/0.5
    elif aoi_id[0:3] == 'Dji':
        zonestring = "38N"
        resolution = 2.5
        upscale_factor = 5

    return zonestring, resolution, upscale_factor


def rectify_gt_dsm(gt_dir, aoi_id):
    #rectify the geo information of the gt dsm for better visualization in QGIS
    gt_dsm_path = gt_dir + aoi_id + '_DSM.tif'
    gt_roi_path = gt_dir + aoi_id + '_DSM.txt'

    dsm_metadata = np.loadtxt(gt_roi_path)
    # read dsm metadata
    xoff, yoff = dsm_metadata[0], dsm_metadata[1]
    xsize, ysize = int(dsm_metadata[2]), int(dsm_metadata[2])
    resolution = dsm_metadata[3]

    dst_transform = Affine.translation(xoff, yoff+xsize*resolution) * Affine.scale(resolution, -resolution)
    gt_dsm_out_path = gt_dsm_path[:-4] +"_scl.tif"
    ScaleDSM(gt_dsm_path, zonestring, gt_dsm_out_path, upscale_factor=1, dst_transform=dst_transform)

    return dsm_out_path, gt_dsm_out_path

def CalcMAE(in_dsm_path, gt_dsm_path, gt_roi_path, gt_seg_path, rdsm_path, rdsm_diff_path, mask_out_path, outputMask=False):
    gt_roi_metadata = np.loadtxt(gt_roi_path)

    diff = dsm_pointwise_diff(in_dsm_path, gt_dsm_path, gt_roi_metadata, gt_mask_path=gt_seg_path,
                                       out_rdsm_path=rdsm_path, out_err_path=rdsm_diff_path)

    isnan = np.isnan(diff)
    if outputMask:
        mask=(isnan==False)
        Image.fromarray(mask).save(mask_out_path)
    nanNb = np.sum(isnan == True)
    totalNb = diff.shape[0]*diff.shape[1]

    return diff, nanNb, totalNb

def cal_rmse_depth(densedepth_file, gt_dir, aoi_id):
    ###########################prepare dsm from dense depth###########################
    dsm_out_path = densedepth_file[:-4] +"_tmp.tif"
    roi_txt = None
    dst_transform = None

    zonestring, resolution, upscale_factor = get_info(aoi_id)
    
    assert os.path.exists(densedepth_file), f"{densedepth_file} not found"
    get_dsm_from_dense_depth(densedepth_file, zonestring, dsm_out_path, resolution, roi_txt=roi_txt, dst_transform=dst_transform)

    dsm_in_path = dsm_out_path
    dsm_out_path = densedepth_file[:-4] +".tif"
    ScaleDSM(dsm_in_path, zonestring, dsm_out_path, upscale_factor=upscale_factor)
    os.remove(dsm_in_path)

    #dsm_out_path, gt_dsm_out_path = preprocess_dsms(densedepth_file, gt_dir, aoi_id)

    ###########################CalcMAE###########################
    gt_roi_path = gt_dir + aoi_id + '_DSM.txt'
    gt_seg_path = None
    gt_dsm_path = gt_dir + aoi_id + '_DSM.tif'

    rdsm_path = dsm_out_path[:-4] + '_aligned.tif'
    rdsm_diff_path = dsm_out_path[:-4] + '_dod.tif'
    mask_out_path = dsm_out_path[:-4] + 'mask.tif'
    outputMask = True

    diff, nanNb, totalNb = CalcMAE(dsm_out_path, gt_dsm_path, gt_roi_path, gt_seg_path, rdsm_path, rdsm_diff_path, mask_out_path, outputMask=outputMask)
    mae = np.nanmean(abs(diff.ravel()))
    print('--->>>MAE: {:.2f}, nan pts number in DSM transformed from depth map: {} out of {} ({:.2f}%)'.format(mae, nanNb, totalNb, (nanNb)*100.0/totalNb))

