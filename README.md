# CSED499_Project

## Directory Structure
- `data` : directory for image datas.
- `data_loader` : codes for loading images from data directory
- `docs` : document directory
- `model` : codes for pytorch models and loading weights
- `trainer`


## Train models
- `python train_face.py` 
- `python train_flower.py` 
- `python train_gtsrb.py` 
model weights are saved in saved/ directory

## Attack Image generation
- `python train_*.py` : train model using transfer leaning
- `python train_attack_*.py` : train attack image using student models generated by transfer learning.