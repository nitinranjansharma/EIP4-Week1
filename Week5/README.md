# Approach taken
## 1. Inception V3 model taken from Keras implementation and trained from scratch
## 2. Edited the Inception V3 model to fecilitate parallel entry point from input image transforming to 4 heads parallel to mixer
## 3. Augmentation technique used were image masking/ and dropout since data was too complex
## 4. Two methods for training were used - a. Training at a stretch to 25 epochs and b. Training multiple times by save and load approach to 150 epochs
# Accuracy given for all below
