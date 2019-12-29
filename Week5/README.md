# Approach taken
## 1. Inception V3 model taken from Keras implementation and trained from scratch
## 2. Edited the Inception V3 model to fecilitate parallel entry point from input image transforming to 4 heads parallel to mixer
## 3. Augmentation technique used were image masking/ and dropout since data was too complex
## 4. Two methods for training were used - a. Training at a stretch to 25 epochs and b. Training multiple times by save and load approach to 150 epochs
# Accuracy given for all below
## 150 Epochs multiple iterations - 'gender_output_acc': 0.8690944881889764, 'image_quality_output_acc': 0.5093503937007874, 'age_output_acc': 0.37450787401574803, 'weight_output_acc': 0.5856299212598425, 'bag_output_acc': 0.6240157480314961, 'footwear_output_acc': 0.6441929133858267, 'pose_output_acc': 0.8174212598425197, 'emotion_output_acc': 0.6338582677165354

## 25 Epochs -  Continuous - 'gender_output_acc': 0.8764763779527559, 'image_quality_output_acc': 0.5447834645669292, 'age_output_acc': 0.3799212598425197, 'weight_output_acc': 0.6117125984251969, 'bag_output_acc': 0.6510826771653543, 'footwear_output_acc': 0.6338582677165354, 'pose_output_acc': 0.812007874015748, 'emotion_output_acc': 0.6904527559055118

## Selected - 25 epochs and model is temp.h5(weights in the repo)

## Architecture is attached as architecture.png
