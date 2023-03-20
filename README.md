# VIT-CIFAR10

## Introduction

This repository contains an implementation of the paper "AN IMAGE IS WORTH 16x16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE" by Dosovitskiy et al. The Vision Transformer (ViT) model leverages the Transformer architecture to achieve state-of-the-art performance in image recognition tasks. In this implementation, focus on the popular CIFAR-10 dataset.

## Folder Structure

Below is the folder structure of this repository:

```
├── config
│   └── config.yaml        # Configuration parameters for the model, dataset, and training
├── data.py                # Load and preprocess the CIFAR-10 dataset
├── models.py              # Define the Vision Transformer model architecture
├── train.py               # Implement the training loop for the model
├── test.py                # Implement the testing loop to evaluate the model's performance
└── utils.py               # Utility functions or classes required for the implementation
```

## Dependency

Our code requires the following libraries:

- [PyTorch](https://pytorch.org/)

```
pip install PyYAML
pip install numpy
pip install matplotlib
```

## Model

![https://user-images.githubusercontent.com/37692743/226306173-d2d5661b-4808-483f-9f19-0678c998f482.png](https://user-images.githubusercontent.com/37692743/226306173-d2d5661b-4808-483f-9f19-0678c998f482.png)

The ViT model architecture consists of the following main components:

1. **Embedding**: The input image is divided into patches and then linearly embedded into a flat vector using a patch embedding layer.
2. **TransformerEncoder**: The embedded patches are passed through a series of Transformer encoder layers, each consisting of a MultiHeadAttention mechanism, LayerNorm, and a feed-forward network with ReLU activation.
3. **MLPHead**: The output of the TransformerEncoder is passed through a final Multi-Layer Perceptron (MLP) head that generates the final class probabilities.

```
Model(
  (embedding): Embedding(
    (devider): Devider()
    (patch_embedding): Linear(in_features=12, out_features=128, bias=True)
  )
  (transformer_encoder): TransformerEncoder(
    (layers): ModuleList(
      (0): ModuleList(
        (0): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (1): MultiHeadAttention(
          (query): Linear(in_features=128, out_features=128, bias=True)
          (key): Linear(in_features=128, out_features=128, bias=True)
          (value): Linear(in_features=128, out_features=128, bias=True)
          (out): Linear(in_features=128, out_features=128, bias=True)
        )
        (2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (3): Sequential(
          (0): Linear(in_features=128, out_features=512, bias=True)
          (1): ReLU()
          (2): Linear(in_features=512, out_features=128, bias=True)
        )
        (4): Dropout(p=0.1, inplace=False)
      )
      (1~11) Same as above (0)
    )
  )
  (mlp_head): MLPHead(
    (fc): Linear(in_features=128, out_features=1000, bias=True)
  )
)
```

## Training and Evaluation

Example of configuration file for learning and evaluation

```
device: cuda
train:
  batch_size: 512
  train_epochs: 40
  loss:
    - crossentropyloss
  optim:
    name: adam
    learning_rate: 0.001
    others:
      betas: [0.9, 0.999]
  lr_scheduler:
    name: multisteplr
    others:
      milestones: [20, 30]
      gamma: 0.1
  model_path: ./pretrained
  progress_path: ./train_progress
  plot_epochs: 5
model:
  device: cuda
  patch_size: 2
  in_channels: 3
  hidden_dim: 128
  num_heads: 4
  feedforward_dim: 512
  num_layers: 12
  dropout_rate: 0.1
  num_patches: 256
  num_classes: 1000
test:
  batch_size: 256
```

### Run Train

Use the code below

```
CONFIGURATION_PATH = './config/config.yaml'

python train.py --config $CONFIGURATION_PATH
```

### Run Evaluation

```
CONFIGURATION_PATH = './config/config.yaml'
PRETRAINED_PATH = './pretrained/model_20.pt'

python test.py --config $CONFIGURATION_PATH --pretrained $PRETRAINED_PATH

```

## Results
The results below are from 40 epochs of training. You can see that it is overfitting. We will analyze this result in the discussion below.
![40epochs](https://user-images.githubusercontent.com/37692743/226308359-1045db2e-4acc-408e-98c3-5fa46dbdd641.png)

|  Set  | Classification Accuracy |
|:-----:|:-----------------------:|
| Train |          95%            |
| Test  |          62%            |

## Discussion
In our experiment, we trained the Vision Transformer (ViT) model from scratch on the CIFAR-10 dataset. The final test set accuracy reached 62%, which is relatively low. we discovered that the model was overfitting during training.

There are a few possible reasons for overfitting in this case:

1. Limited dataset size: CIFAR-10 is a relatively small dataset with only 60,000 images, which may not be sufficient to train a complex model like ViT from scratch. The original ViT paper utilized large-scale datasets such as ImageNet for pretraining, which helped the model learn more general features.

2. Model complexity: The ViT model has a large number of parameters, making it prone to overfitting, especially when trained on a small dataset.

3. Insufficient regularization: Regularization techniques such as data augmentation, dropout, or weight decay can improve generalization. If these techniques are not utilized or are insufficient, the model is more likely to overfit.

To improve the performance and generalization of the ViT model on the CIFAR-10 dataset, we recommend the following approaches:

- Use a pre-trained ViT model on a larger dataset (e.g., ImageNet) and fine-tune it on CIFAR-10, instead of training from scratch.
- Experiment with different regularization techniques, such as data augmentation, dropout, or weight decay, to reduce overfitting.
- Employ early stopping based on validation set performance to prevent the model from learning noise in the training data.

By addressing the overfitting issue, we can potentially achieve better accuracy on the test set and create a more robust model for image classification tasks.