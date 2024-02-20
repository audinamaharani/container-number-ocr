# ContainerNumber-OCR

## Disclaimer
This repository has been developed as an extension of the work found in the [ContainerNUmber-OCR](https://github.com/lbf4616/ContainerNumber-OCR) repository.

 __Detection:__  Pixel_link + MobileNet_v2  
 __Recognition:__  LSTM  
 __Speed:__  RTX 2080Ti 200ms/image(1920*1080)  
 
![Sample](https://github.com/lbf4616/ContainerNumber-OCR/blob/master/Sample.png)
 
## Quick Start Demo
1. Download the pretrained model from [Google Drive](https://drive.google.com/open?id=18IGl5jOsUX4S6fKLHlw41JXEn4RRxIIF)  
2. Install requirements  
```
conda create --name cnocr python=3.6
conda activate cnocr
pip install -r requirements.txt
```
3. run inference
```
python containernumber_test_ckpt.py
```

## Dataset
The dataset used for training and evaluation can be accessed via [Google Drive]()

## Evaluation
Evaluate the model's performance by using the `evaluate.py` script. Specify the input and label paths using `-i` for input and `-l` for labels, respectively:
```
python evaluate.py -i /path/to/your/data -l /path/to/your/labels
```

## Convert to Openvino Model
```
python tools/export.py \
  --checkpoint ckpt/recognition_v/model_all.ckpt-146000 \
  --output_dir recognition_v_ov
```
