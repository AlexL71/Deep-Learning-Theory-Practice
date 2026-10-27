# Flickr8k data for the image-captioning section

The one-to-many part of `ch08_rnn_lstm_gru_image_captioning.ipynb` trains a caption model on **Flickr8k** (8,091 images, 5 captions each). The dataset is not stored in this repository.

## 1. Download

Download the Flickr8k dataset from Kaggle: <https://www.kaggle.com/datasets/adityajn105/flickr8k>

Unzip it so this folder looks like:

```
lab08_recurrent_networks/
└── archive/
    ├── Images/          # 8,091 .jpg files
    └── captions.txt     # "image,caption" per line
```

## 2. Extract image features

The notebook does not run VGG16 on raw images. It loads 512-dimensional VGG16 features from `archive/features.npy`. Create that file once, from this folder:

```bash
cd lab08_recurrent_networks
python extract_features.py
```

This runs VGG16 (ImageNet weights, global average pooling) over all 8,091 images. Expect a few minutes on a GPU and much longer on a CPU.

## 3. Run the notebook

The notebook then writes `archive/model_caption.keras` and `archive/tokenizer.pkl`, and reloads them for the caption examples at the end. Everything in `archive/` is ignored by git.
