DataSet

mnist - by torch
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    (28, 28, 1)

bhmsds - https://github.com/wblachowski/bhmsds (has hyphen)
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, *, -, +, /, w, x, y, z
    need to invert
    (28, 28, 3)

curated - 
    ASCII Code [33 ~ 91, 93 ~ 126]
    (64, 64, 3)

archive2 - https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset/data?select=Img (start with img)
    [0 ~ 9, A ~ Z, a ~ z]
    need to invert
    (900, 1200, 3)

archive3 - https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols/data (has )
    [-, ',', !, (, ), +, 0, ]
    need to invert
    (45, 45, 3)

    
*****NOT USE*****
archive5 - https://www.kaggle.com/datasets/clarencezhao/handwritten-math-symbol-dataset
    []


---

References
    * https://www.mdpi.com/1999-4893/15/4/129

---

============================================================================================================================================
Layer (type:depth-idx)                   Input Shape               Output Shape              Param #                   Trainable
============================================================================================================================================
PaperCNN                                 [1, 1, 28, 28]            [1, 10]                   --                        True
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

--- 

[Info] Test Results: Accuracy: 98.13%
[Info] Spent Time: 1065.3427 seconds

[Info] Test Results: Accuracy: 84.6%
[Info] Spent Time: 2814.9417 seconds

[Info] Test Results: Accuracy: 99.32%
[Info] Spent Time: 3059.3362 seconds

[Info] Spent Time: 6939.6304 seconds