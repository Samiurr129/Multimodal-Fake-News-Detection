# Multimodal fake news detection

A PyTorch pipeline that classifies a news item as real or fake from **both its text and its image**. A BERT encoder reads the text, a small CNN reads the image, and the two feature vectors are joined before classification.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)
![Licence](https://img.shields.io/badge/licence-MIT-green)

## Model

```
text  -> BERT ([CLS] token, 768) -> Linear 256 -+
                                                 +-> concat 512 -> dropout -> Linear 2 (real / fake)
image -> 224x224 -> CNN (2 conv + pool) -> 256 -+
```

- Text: any Hugging Face BERT-style model (`bert-base-uncased` by default), fine-tuned end to end.
- Image: a two-layer CNN trained from scratch.
- Training: AdamW, cross-entropy loss, stratified 80/20 train/test split.
- Evaluation: precision, recall and F1 for the fake class, plus a full classification report.

## Data

The model needs a CSV with three columns:

| Column | Meaning |
| --- | --- |
| `text` | Headline or article text |
| `image_path` | Path to the image, absolute or relative to the CSV's folder |
| `label` | 0 = real, 1 = fake |

Public multimodal datasets include [Fakeddit](https://github.com/entitize/Fakeddit) and FakeNewsNet. The datasets are not included here, because of their size and licence terms.

## Run it

```bash
git clone https://github.com/Samiurr129/Multimodal-Fake-News-Detection.git
cd Multimodal-Fake-News-Detection
pip install -r requirements.txt

# 1. Check the pipeline end to end on a tiny synthetic dataset (about a minute on CPU)
python make_demo_data.py
python train.py --data demo_data/demo.csv --bert-model google/bert_uncased_L-2_H-128_A-2 --epochs 1 --max-length 32

# 2. Train on a real dataset
python train.py --data data/fakenews_dataset.csv --epochs 5 --batch-size 16
```

The demo data is random, so its scores mean nothing. It only proves the code runs.

## Status

The pipeline is complete and tested end to end on demo data. No results are reported yet, because the model has not been trained on a full public dataset.

## Next steps

- Train and report results on Fakeddit
- Replace the small CNN with a pretrained ResNet or ViT image encoder
- Compare text-only, image-only and multimodal models (ablation)
- Serve the model with a small FastAPI endpoint

## Author

**Samiur Rahman** · [LinkedIn](https://www.linkedin.com/in/samiur-rahman29/) · [GitHub](https://github.com/Samiurr129)
