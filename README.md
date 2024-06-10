# Handwritten Character and Symbol Recognition

This project focuses on the training and evaluation of convolutional neural networks (CNNs) to recognize handwritten characters and symbols. Various datasets are utilized, including MNIST, BHMSDS, and a curated dataset, each containing different types of handwritten data. The model architecture used for this task is a custom CNN called `CNN`. Below are the details of the datasets, model structure, and the results from the training and evaluation process.

### DataSet Information

#### 1. **MNIST (by torch)**
- **Classes**: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`
- **Image Shape**: `(28, 28, 1)`

#### 2. **BHMSDS**
- **Source**: [GitHub - wblachowski/bhmsds](https://github.com/wblachowski/bhmsds)
- **Classes**: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, *, -, +, /, w, x, y, z`
- **Preprocessing**: Need to invert images
- **Image Shape**: `(28, 28, 3)`

#### 3. **Curated Dataset**
- **Classes**: ASCII Code `[33 ~ 91, 93 ~ 126]`
- **Image Shape**: `(64, 64, 3)`

#### 4. **Archive3**
- **Source**: [Kaggle - Handwritten Math Symbols](https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols/data)
- **Classes**: `[,-, ',', !, (, ), +, 0]`
- **Preprocessing**: Need to invert images
- **Image Shape**: `(45, 45, 3)`

#### 5. **Not Used Dataset**
- **Source**: [Kaggle - Handwritten Math Symbol Dataset](https://www.kaggle.com/datasets/clarencezhao/handwritten-math-symbol-dataset)
- **Source**: [Kaggle - English Handwritten Characters Dataset](https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset/data?select=Img)

### Model Structure

#### **NumberModel**
```plaintext
============================================================================================================================================
Layer (type:depth-idx)                   Input Shape               Output Shape              Param #                   Trainable
============================================================================================================================================
NumberModel                              [1, 1, 28, 28]            [1, 10]                   --                        True
├─Conv2d: 1-1                            [1, 1, 28, 28]            [1, 32, 26, 26]           320                       True
├─Conv2d: 1-2                            [1, 32, 26, 26]           [1, 64, 26, 26]           18,496                    True
├─MaxPool2d: 1-3                         [1, 64, 26, 26]           [1, 64, 13, 13]           --                        --
├─Conv2d: 1-4                            [1, 64, 13, 13]           [1, 128, 13, 13]          73,856                    True
├─MaxPool2d: 1-5                         [1, 128, 13, 13]          [1, 128, 6, 6]            --                        --
├─Dropout2d: 1-6                         [1, 128, 6, 6]            [1, 128, 6, 6]            --                        --
├─Conv2d: 1-7                            [1, 128, 6, 6]            [1, 256, 6, 6]            295,168                   True
├─MaxPool2d: 1-8                         [1, 256, 6, 6]            [1, 256, 3, 3]            --                        --
├─Dropout2d: 1-9                         [1, 256, 3, 3]            [1, 256, 3, 3]            --                        --
├─Linear: 1-10                           [1, 2304]                 [1, 64]                   147,520                   True
├─Linear: 1-11                           [1, 64]                   [1, 10]                   650                       True
============================================================================================================================================
Total params: 536,010
Trainable params: 536,010
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 35.98
============================================================================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.77
Params size (MB): 2.14
Estimated Total Size (MB): 2.91
============================================================================================================================================
```

#### **CharacterModel**
```plaintext
============================================================================================================================================
Layer (type:depth-idx)                   Input Shape               Output Shape              Param #                   Trainable
============================================================================================================================================
CharacterModel                           [1, 1, 64, 64]            [1, 10]                   --                        True
├─Conv2d: 1-1                            [1, 1, 64, 64]            [1, 32, 62, 62]           320                       True
├─MaxPool2d: 1-2                         [1, 32, 62, 62]           [1, 32, 31, 31]           --                        --
├─Conv2d: 1-3                            [1, 32, 31, 31]           [1, 64, 29, 29]           18,496                    True
├─MaxPool2d: 1-4                         [1, 64, 29, 29]           [1, 64, 14, 14]           --                        --
├─Conv2d: 1-5                            [1, 64, 14, 14]           [1, 128, 14, 14]          73,856                    True
├─MaxPool2d: 1-6                         [1, 128, 14, 14]          [1, 128, 7, 7]            --                        --
├─Conv2d: 1-7                            [1, 128, 7, 7]            [1, 256, 7, 7]            295,168                   True
├─MaxPool2d: 1-8                         [1, 256, 7, 7]            [1, 256, 3, 3]            --                        --
├─Linear: 1-9                            [1, 2304]                 [1, 256]                  590,080                   True
├─Linear: 1-10                           [1, 256]                  [1, 10]                   2,570                     True
============================================================================================================================================
Total params: 980,490
Trainable params: 980,490
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 46.32
============================================================================================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 1.72
Params size (MB): 3.92
Estimated Total Size (MB): 5.66
============================================================================================================================================
```

#### **SymbolModel**
```plaintext
============================================================================================================================================
Layer (type:depth-idx)                   Input Shape               Output Shape              Param #                   Trainable
============================================================================================================================================
SymbolModel                              [1, 1, 45, 45]            [1, 10]                   --                        True
├─Conv2d: 1-1                            [1, 1, 45, 45]            [1, 32, 43, 43]           320                       True
├─Conv2d: 1-2                            [1, 32, 43, 43]           [1, 64, 43, 43]           18,496                    True
├─MaxPool2d: 1-3                         [1, 64, 43, 43]           [1, 64, 21, 21]           --                        --
├─Conv2d: 1-4                            [1, 64, 21, 21]           [1, 128, 21, 21]          73,856                    True
├─MaxPool2d: 1-5                         [1, 128, 21, 21]          [1, 128, 10, 10]          --                        --
├─Dropout2d: 1-6                         [1, 128, 10, 10]          [1, 128, 10, 10]          --                        --
├─Conv2d: 1-7                            [1, 128, 10, 10]          [1, 256, 10, 10]          295,168                   True
├─MaxPool2d: 1-8                         [1, 256, 10, 10]          [1, 256, 5, 5]            --                        --
├─Dropout2d: 1-9                         [1, 256, 5, 5]            [1, 256, 5, 5]            --                        --
├─Linear: 1-10                           [1, 6400]                 [1, 64]                   409,664                   True
├─Linear: 1-11                           [1, 64]                   [1, 10]                   650                       True
============================================================================================================================================
Total params: 798,154
Trainable params: 798,154
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 97.29
============================================================================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 2.08
Params size (MB): 3.19
Estimated Total Size (MB): 5.28
============================================================================================================================================
```

### Training and Evaluation Results

#### **Number Model**
- **Test Accuracy**: 98.13%
- **Total Time Spent**: 1137.2781 seconds

#### **Character Model**
- **Test Accuracy**: 75.91%%
- **Total Time Spent**: 1607.1807 seconds

#### **Symbol Model**
- **Test Accuracy**: 99.4%
- **Total Time Spent**: 1727.4221 seconds

#### **Overall Time Spent**
- **Total Time Spent**: 4471.8809 seconds



### References
- [MDPI - Algorithms Journal](https://www.mdpi.com/1999-4893/15/4/129)