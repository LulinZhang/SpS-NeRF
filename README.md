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

2. Create virtualenv `spsnerf`
```
conda init
bash -i setup_spsnerf_env.sh
```

### Optional
If you want to prepare the dataset yourself, you'll need to create virtualenv `ba`:
```
conda init
bash -i setup_ba_env.sh
```

## 1. Prepare dataset
You can skip this step and directly download the [DFC2019 dataset AOI 214](https://drive.google.com/file/d/1LXfkxe_d3WSVgxK5y8q4Si-sUF6Fvv-R/view?usp=sharing).

*You need to prepare a directory `ProjDir` to place the dataset.*

### 1.1. Refine RPC with bundle adjustment
Please use command `conda activate ba` to get into the `ba` environment for this step.
```
BaseDir=/home/LZhang/Documents/CNESPostDoc/SpSNeRFProj/input_prepare_data/
aoi_id=JAX_214
DataDir="$BaseDir"DFC2019/
OutputDir="$BaseDir"
python3 create_satellite_dataset.py --aoi_id "$aoi_id" --dfc_dir "$DataDir" --output_dir "$OutputDir" 
```

*Please replace the values from first to third lines in the above script to your own value.*

In your `DataDir`, it should contain the RGB images, ground truth DSM and other text files to indicate necessary information. Please refer to [our example](https://drive.google.com/file/d/1kVuEKONJamt1R9caxY-zJFVQXgbiQDsM/view?usp=sharing) for file organization.

### 1.2. Generate dense depth
You can skip this step and directly download the [Dense depth of DFC2019 dataset AOI 214](https://drive.google.com/file/d/1L7PmSCaNvQGtk6mNyfufp3z8hbzSNiQM/view?usp=sharing) and put it in your `TxtDenseDir`.

#### Option 1: Use software MicMac
In our experiments, this step is done with the free, open-source photogrammetry software `MicMac`. You need to install MicMac following [this websit](https://github.com/micmacIGN/micmac).
```
BaseDir=/home/LZhang/Documents/CNESPostDoc/SpSNeRFProj/input_prepare_data/
aoi_id=JAX_214
DataDir="$BaseDir"DFC2019/
RootDir="$BaseDir"JAX_214_2_imgs/
TxtDenseDir="$RootDir"dataset"$aoi_id"/root_dir/crops_rpcs_ba_v2/"$aoi_id"/DenseDepth_ZM4/
MicMacDenseDir="$RootDir"DenseDepth/
CodeDir=/home/LZhang/Documents/CNESPostDoc/SpSNeRFProj/code/SpS-NeRF/

mkdir "$MicMacDenseDir"
mkdir "$TxtDenseDir"

#copy the images and refined rpc parameters
for line in `cat "$DataDir"train.txt`
do
	img_name=${line%.*}
	cp "$DataDir"RGB/"$aoi_id"/"$img_name".tif "$MicMacDenseDir""$img_name".tif
	cp "$RootDir"ba_files/rpcs_adj/"$img_name".rpc_adj "$MicMacDenseDir""$img_name".txt
done
cp "$DataDir"WGS84toUTM.xml "$MicMacDenseDir"WGS84toUTM.xml
cd "$MicMacDenseDir"

#convert rpc to the MicMac format
mm3d Convert2GenBundle "(.*).tif" "\$1.txt" RPC-d0-adj ChSys=WGS84toUTM.xml Degre=0

for line in `cat "$DataDir"train.txt`
do
	img_name=${line%.*}
	#generate dense depth in tif format
	mm3d Malt GeomImage ".*tif" RPC-d0-adj Master="$img_name".tif SzW=1 Regul=0.05 NbVI=2 ZoomF=4 ResolTerrain=1 EZA=1 DirMEC=MM-"$img_name"/
	#convert dense depth tif to txt format
	mm3d TestLib GeoreferencedDepthMap MM-"$img_name" "$img_name".tif Ori-RPC-d0-adj OutDir="$TxtDenseDir" Mask=1 Scale=4
done

cd "$CodeDir"
#Transform 3D points from UTM to geocentric coordinates.
python3 utm_to_geocentric.py --file_dir "$TxtDenseDir"
```

*Please replace the values from first to third, and sixth lines in the above script to your own value.*

#### Option 2: Use other software
It is also possible if you prefer to use other software, just make sure your final result is organized this way:
- `TxtDenseDir`
  - `ImageName_2DPts.txt`: 2D coordinate in image frame for the pixels with valid depth value. The first line is width, and the second line is height.
  - `ImageName_3DPts.txt`: 3D coordinate in UTM for the pixels with valid depth value.
  - `ImageName_3DPts_ecef.txt`: 3D coordinate in geocentric coordinates for the pixels with valid depth value.
  - `ImageName_Correl.txt`: correlation score for the pixels with valid depth value.

Each image `ImageName` corresponds to four txt files as displayed below.

## 2. Train SpS-NeRF
Please use command `conda activate spsnerf` to get into the `spsnerf` environment for this step.
```
aoi_id=JAX_214
inputdds=DenseDepth_ZM4
n_importance=0
ds_lambda=1
stdscale=1
ProjDir=/gpfs/users/lzhang/SpS-NeRF_test/
exp_name=SpS_output"$aoi_id"-"$inputdds"-FnMd"$n_importance"-ds"$ds_lambda"-"$stdscale"
Output="$ProjDir"/"$exp_name"
rm -r "$Output"
mkdir "$Output"
cp Sh-SpS-Train-JAX_---_2imgs.sh "$Output"/.    

python3 main.py --aoi_id "$aoi_id" --model sps-nerf --exp_name "$exp_name" --root_dir "$ProjDir"/dataset"$aoi_id"/root_dir/crops_rpcs_ba_v2/"$aoi_id"/ --img_dir "$ProjDir"/dataset"$aoi_id"/"$aoi_id"/RGB-crops/"$aoi_id"/ --cache_dir "$Output"/cache_dir/crops_rpcs_ba_v2/"$aoi_id" --gt_dir "$ProjDir"/dataset"$aoi_id"/"$aoi_id"/Truth --logs_dir "$Output"/logs --ckpts_dir "$Output"/ckpts --inputdds "$inputdds" --gpu_id 0 --img_downscale 1 --max_train_steps 30000 --lr 0.0005 --sc_lambda 0 --ds_lambda "$ds_lambda" --ds_drop 1 --n_importance "$n_importance" --stdscale "$stdscale" --guidedsample --mapping    
```

*Please replace the value of `ProjDir` in the second line in the above script to your own `ProjDir`.*

## 3. Test SpS-NeRF
### 3.1. Render novel views
Please use command `conda activate spsnerf` to get into the `spsnerf` environment for this step.
```
Output=/gpfs/users/lzhang/SpSNeRFProj/DFCDataClean_2imgs/SpS_outputJAX_214-DenseDepth_ZM4-FnMd0-ds1-1/
logs_dir="$Output"/logs
run_id=SpS_outputJAX_214-DenseDepth_ZM4-FnMd0-ds1-1
output_dir="$Output"/eval_spsnerf
epoch_number=28

python3 eval.py --run_id "$run_id" --logs_dir "$logs_dir" --output_dir "$output_dir" --epoch_number "$epoch_number" --split val
```

*Please replace the value of `Output`, `run_id`, `output_dir` and `epoch_number` in the above script to your own settings.*

### 3.2. Generate DSM (Digital Surface Model)
Please use command `conda activate spsnerf` to get into the `spsnerf` environment for this step.
```
Output=/gpfs/users/lzhang/SpSNeRFProj/DFCDataClean_2imgs/SpS_outputJAX_214-DenseDepth_ZM4-FnMd0-ds1-1/
logs_dir="$Output"/logs
run_id=SpS_outputJAX_214-DenseDepth_ZM4-FnMd0-ds1-1
output_dir="$Output"/create_spsnerf_dsm
epoch_number=28

python3.6 ../../code/SpS-NeRF/create_dsm.py --run_id "$run_id" --logs_dir "$logs_dir" --output_dir "$output_dir" --epoch_number "$epoch_number"
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
