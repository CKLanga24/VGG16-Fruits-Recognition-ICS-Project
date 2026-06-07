VGG16-Fruits-Recognition-ICS-Project

### Contribution of Team Members ###

CK LANGA

* Dataset preparation and splitting (70% train, 15% validation, 15% test).
* VGG16 model setup and coding.
* GitHub repository creation and management.

Aryan Kumar

* Creating and Implimenting Custom loss function.

Mohit Kumar

* Model training on 15 fruit classes (5 epochs per training run).
* Result generation and reporting.

### Model Architecture ###

This project uses the VGG16 Convolutional Neural Network (CNN) for fruit image classification.

* Pre-trained VGG16 model with ImageNet weights.
* Top classification layers of VGG16 were removed (`include_top=False`).
* All VGG16 layers were frozen to utilize transfer learning.
* Input image size: 224 × 224 × 3.
* A custom classification head was added:

  * Flatten layer
  * Dense layer with 256 neurons and ReLU activation
  * Dropout layer (0.5) to reduce overfitting
  * Output Dense layer with Softmax activation for multi-class fruit classification
* Custom loss function combining:

  * Label smoothing
  * Focal loss
  * L2 regularization
* Optimizer: Adam
* Evaluation metric: Accuracy

Architecture Flow:

Input Image (224×224×3)
→ VGG16 Feature Extractor (Frozen)
→ Flatten
→ Dense (256, ReLU)
→ Dropout (0.5)
→ Dense (Number of Fruit Classes, Softmax)
→ Predicted Fruit Class

### Final Result ###
  * Test Accuracy: 0.9810820817947388 (98.108 %)
  * Test Loss: 0.04125632345676422
  * Training time : 12 hours (Approximate)

### Conclusion ###
  The high accuracy was achieved due to the large dataset, clear visual differences between fruit classes, and the use of transfer learning with VGG16. Training for 5 epochs was sufficient for the model to learn the relevant features.






