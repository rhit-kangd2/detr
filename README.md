**DE⫶TR**: End-to-End Object Detection with Transformers
========

Original GitHub repo: https://github.com/facebookresearch/detr

# Set-up
First, clone the repository locally. Then, install PyTorch 1.5+ and torchvision 0.6+:
```
conda install -c pytorch pytorch torchvision
```
Install pycocotools (for evaluation on COCO) and scipy (for training):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
WARNING: Some packages may use the outdated "np.float" type. If this error occurs, change them to "float".

## Training
To train baseline DETR on a single node with 8 gpus for 1 epoch run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path ./coco 
```
A single epoch takes around 45 minutes on Rose's gebru server using 8 NVDA GPUs.

## Evaluation
To evaluate DETR R50 on COCO val5k with a single GPU run:
```
python main.py --batch_size 2 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path ./coco
```

## Demo
To demo the models,
1. Place the input image into ./sample_images/ and name it sample.jpg.
2. Run demo.py to run the trained model, or run demo_pretrained.py to run Facebook's pretrained model.
3. View the classification and confidence results in the terminal and labeled output in ./sample_output/.
