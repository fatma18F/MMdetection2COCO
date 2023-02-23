## Convert MMdetection inference prediction to COCO format 
about [MMdetection](https://storage.googleapis.com/openimages/web/index.html):
MMdetection is an open-source library containing many popular and state-of-the-art object detection models.

### PREREQUISITES
***Step 1.*** Create a conda environment and activate it.
```
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch torchvision -c pytorch
```
***Step 2.*** Install MMCV using MIM.
```
pip install -U openmim
mim install mmcv-full
pip install mmdet

```
***Step 3.*** Install MMDetection.
```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```
***Step 4:*** Install needed libraries:
```
pip install imagesize
pip install tqdm
```
***Step 5:*** Download config and checkpoint files of the selected model from mmdet
For example for yolov3 run :
```
mim download mmdet --config yolov3_mobilenetv2_320_300e_coco --dest .
```

### Functionality

-  `image_predictions2coco.py` runs inference on the specified image path and convert the model predictions into the list/dict based format of [MS Coco annotations](http://cocodataset.org/#format-data) and store them as a .json file in the same folder.
  
***parameters*** : specify image path under `dir_path`, model configuation file path under `config_file` and model weights path under `checkpoint_file`


- `tarsierDataset_converor.py` fellows Tarsier images structure and runs inference on the specified number of images from in `/mnt/NAS_Backup/Datasets/Tarsier_Main_Dataset/Images`. The default `nb_images=1000` and convert the model predictions into the list/dict based format of [MS Coco annotations](http://cocodataset.org/#format-data) and store them as a .json file in the same folder.

***parameters*** : specify image path under `img_prefix`,annotation file under `ann_file `,model configuation file path under `config_file` and model weights path under `checkpoint_file`

