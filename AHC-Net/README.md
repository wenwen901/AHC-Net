- You need to Clone this repository:
```bash
git https://github.com/wenwen901/AHC-Net
cd AHC-Net
```

### Datasets
1) ISIC 2018 - [Link](https://challenge.isic-archive.com/data/)
2) BUSI - [Link](https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset)
3) GLAS - [Link](https://websignon.warwick.ac.uk/origin/slogin?shire=https%3A%2F%2Fwarwick.ac.uk%2Fsitebuilder2%2Fshire-read&providerId=urn%3Awarwick.ac.uk%3Asitebuilder2%3Aread%3Aservice&target=https%3A%2F%2Fwarwick.ac.uk%2Ffac%2Fcross_fac%2Ftia%2Fdata%2Fglascontest&status=notloggedin)
4) CVC-ClinicDB - [Link](https://www.kaggle.com/datasets/balraj98/cvcclinicdb)


### Data Format
- Make sure to put the files as the following structure. For binary segmentation, just use folder 0.
```
inputs
└── <dataset name>
    ├── images
    |   ├── 001.png
    │   ├── 002.png
    │   ├── 003.png
    │   ├── ...
    |
    └── masks
        └── 0
            ├── 001.png
            ├── 002.png
            ├── 003.png
            ├── ...
```


### Training and Validation
- Train the model.
```
python train.py
```
- Evaluate.
```
python val.py
```

