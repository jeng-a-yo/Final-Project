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
    https://www.mdpi.com/1999-4893/15/4/129