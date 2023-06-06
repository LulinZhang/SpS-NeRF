# SpS-NeRF

### [Project page](https://erupnik.github.io/SparseSatNerf.html)

### [SparseSat-NeRF: Dense Depth Supervised Neural Radiance Fields for Sparse Satellite Images](https://blank)
*[Lulin Zhang](https://scholar.google.com/citations?user=tUebgRIAAAAJ&hl=fr&oi=ao),
[Ewelina Rupnik](https://erupnik.github.io/)*

![](documents/teaser.png)


## Setup
### Compulsory
The following steps are compulsory for running this repository:
1. Clone the git repository 
```
git clone https://github.com/LulinZhang/SpS-NeRF.git
```

2. Create virtualenv `satnerf`
```
conda init
bash -i setup_satnerf_env.sh
```

### Optional
If you want to prepare the dataset yourself, you'll need to create virtualenv `ba`:
```
conda init
bash -i setup_ba_env.sh
```

## 1. Prepare dataset
You can skip this step and directly download the [DFC2019 dataset AOI 214](https://drive.google.com/file/d/1LXfkxe_d3WSVgxK5y8q4Si-sUF6Fvv-R/view?usp=sharing).

### Organize your data


### Refine RPC with bundle adjustment
```
conda activate ba
DataDir=/home/LZhang/Documents/CNESPostDoc/SatNeRFProj/input_prepare_data/DFC2019/
OutputDir=/home/LZhang/Documents/CNESPostDoc/SatNeRFProj/input_prepare_data/
aoi_id=JAX_214
python3 create_satellite_dataset.py --aoi_id "$aoi_id" --dfc_dir "$DataDir" --output_dir "$OutputDir" 
```

*Please replace the value of `DataDir` and `OutputDir` in the second and third lines in the above script to your own value.*

### Generate dense depth
In our experiments, this step is done with the free, open-source photogrammetry software `MicMac`, you need to install MicMac following [this websit](https://github.com/micmacIGN/micmac).

```
aoi_id=JAX_214
DataDir=/home/LZhang/Documents/CNESPostDoc/SatNeRFProj/input_prepare_data/DFC2019/
RootDir=/home/LZhang/Documents/CNESPostDoc/SatNeRFProj/input_prepare_data/JAX_214_2_imgs/
MicMacDenseDir="$RootDir"DenseDepth/
TxtDenseDir="$RootDir"dataset"$aoi_id"/root_dir/crops_rpcs_ba_v2/"$aoi_id"/DenseDepth_ZM4/
CodeDir=/home/LZhang/Documents/CNESPostDoc/SatNeRFProj/code/SpS-NeRF/

mkdir "$MicMacDenseDir"
mkdir "$TxtDenseDir"

#copy the images and refined rpc parameters
cp "$RootDir"dataset"$aoi_id"/DFC2019/RGB-crops/"$aoi_id"/"$aoi_id"_009_RGB.tif "$MicMacDenseDir""$aoi_id"_009_RGB.tif
cp "$RootDir"dataset"$aoi_id"/DFC2019/RGB-crops/"$aoi_id"/"$aoi_id"_010_RGB.tif "$MicMacDenseDir""$aoi_id"_010_RGB.tif
cp "$RootDir"ba_files/rpcs_adj/"$aoi_id"_009_RGB.rpc_adj "$MicMacDenseDir""$aoi_id"_009_RGB.txt
cp "$RootDir"ba_files/rpcs_adj/"$aoi_id"_010_RGB.rpc_adj "$MicMacDenseDir""$aoi_id"_010_RGB.txt
cp "$DataDir"WGS84toUTM.xml "$MicMacDenseDir"WGS84toUTM.xml
cd "$MicMacDenseDir"

#convert rpc to the MicMac format
mm3d Convert2GenBundle "(.*).tif" "\$1.txt" RPC-d0-adj ChSys=WGS84toUTM.xml Degre=0

#generate dense depth in tif format
mm3d Malt GeomImage "JAX.*tif" RPC-d0-adj Master="$aoi_id"_010_RGB.tif SzW=1 Regul=0.05 NbVI=2 ZoomF=4 ResolTerrain=1 EZA=1 DirMEC=MM-"$aoi_id"_010_RGB_ZM4/ 
mm3d Malt GeomImage "JAX.*tif" RPC-d0-adj Master="$aoi_id"_009_RGB.tif SzW=1 Regul=0.05 NbVI=2 ZoomF=4 ResolTerrain=1 EZA=1 DirMEC=MM-"$aoi_id"_009_RGB_ZM4/ 

#convert dense depth tif to txt format
mm3d TestLib GeoreferencedDepthMap MM-"$aoi_id"_009_RGB_ZM4 "$aoi_id"_009_RGB.tif Ori-RPC-d0-adj OutDir="$TxtDenseDir" Mask=1 Scale=4
mm3d TestLib GeoreferencedDepthMap MM-"$aoi_id"_010_RGB_ZM4 "$aoi_id"_010_RGB.tif Ori-RPC-d0-adj OutDir="$TxtDenseDir" Mask=1 Scale=4

cd "$CodeDir"
python3 utm_to_ecef.py --file_dir "$TxtDenseDir"
```

*You need to prepare a directory `ProjDir` to place the dataset.*

## 2. Train SpS-NeRF
```
conda activate satnerf
ProjDir=/gpfs/users/lzhang/SpS-NeRF_test/
exp_name=SpS_output"$aoi_id"-"$inputdds"-FnMd"$n_importance"-ds"$ds_lambda"-"$stdscale"
Output="$ProjDir"/"$exp_name"
aoi_id=JAX_214
inputdds=DenseDepth_ZM4
n_importance=0
ds_lambda=1
stdscale=1
rm -r "$Output"
mkdir "$Output"
cp Sh-SpS-Train-JAX_---_2imgs.sh "$Output"/.    

python3 main.py --aoi_id "$aoi_id" --model sps-nerf --exp_name "$exp_name" --root_dir "$ProjDir"/dataset"$aoi_id"/root_dir/crops_rpcs_ba_v2/"$aoi_id"/ --img_dir "$ProjDir"/dataset"$aoi_id"/DFC2019/RGB-crops/"$aoi_id"/ --cache_dir "$Output"/cache_dir/crops_rpcs_ba_v2/"$aoi_id" --gt_dir "$ProjDir"/dataset"$aoi_id"/DFC2019/Truth --logs_dir "$Output"/logs --ckpts_dir "$Output"/ckpts --inputdds "$inputdds" --gpu_id 0 --img_downscale 1 --max_train_steps 30000 --lr 0.0005 --sc_lambda 0 --ds_lambda "$ds_lambda" --ds_drop 1 --n_importance "$n_importance" --stdscale "$stdscale" --guidedsample --mapping    
```

*Please replace the value of `ProjDir` in the second line in the above script to your own `ProjDir`.*

## 3. Test SpS-NeRF
### Render novel views
```
conda activate satnerf
Output=/gpfs/users/lzhang/SatNeRFProj/DFCDataClean_2imgs/SpS_outputJAX_214-DenseDepth_ZM4-FnMd0-ds1-1/
logs_dir="$Output"/logs
run_id=SpS_outputJAX_214-DenseDepth_ZM4-FnMd0-ds1-1
output_dir="$Output"/eval_satnerf
epoch_number=28

python3 eval_satnerf.py --run_id "$run_id" --logs_dir "$logs_dir" --output_dir "$output_dir" --epoch_number "$epoch_number" --split val
```

*Please replace the value of `Output`, `run_id`, `output_dir` and `epoch_number` in the above script to your own settings.*

### Generate DSM (Digital Surface Model)
```
conda activate satnerf
Output=/gpfs/users/lzhang/SatNeRFProj/DFCDataClean_2imgs/SpS_outputJAX_214-DenseDepth_ZM4-FnMd0-ds1-1/
logs_dir="$Output"/logs
run_id=SpS_outputJAX_214-DenseDepth_ZM4-FnMd0-ds1-1
output_dir="$Output"/create_satnerf_dsm
epoch_number=28

python3.6 ../../code/SpS-NeRF/create_satnerf_dsm.py --run_id "$run_id" --logs_dir "$logs_dir" --output_dir "$output_dir" --epoch_number "$epoch_number"
```

*Please replace the value of `Output`, `run_id`, `output_dir` and `epoch_number` in the above script to your own settings.*


### Acknowledgements
We thank [satnerf](https://github.com/centreborelli/satnerf) and [dense_depth_priors_nerf](https://github.com/barbararoessle/dense_depth_priors_nerf), from which this repository borrows code. 

## Citation
If you find this code or work helpful, please cite:
```
@article{zhang2023spsnerf,
   author = {Lulin Zhang and Ewelina Rupnik},
   title = {SparseSat-NeRF: Dense Depth Supervised Neural Radiance Fields for Sparse Satellite Images},
   journal = {ISPRS Annals},
   year = {2023}
}
```
