# Phishing URL Behaviour Classification

LearnDepth capstone project by Abdul Rehman.

This project predicts whether a URL is phishing (0) or benign (1) using characteristics of the URL text. It does not open the website.

## Dataset

Dataset: [Malicious URLs dataset by sid321axn](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset).

This project uses only the phishing and benign categories. The CSV file is stored at `candidate_data/malicious_phish.csv`. See `DATASET_CITATION.md` for the source details.

## How it works

1. `prepare_data.py` cleans the URLs and labels.
2. `features.py` extracts six numbers, including URL length, digit ratio, and hostname characteristics.
3. `train_model.py` compares models, evaluates the chosen Decision Tree, and saves it in `models/`.
4. `app.py` uses that saved model to classify a URL entered in the Streamlit app.

The training and test sets have no shared hostnames.

## Files

- `eda.ipynb` — exploratory data analysis notebook
- `eda_final.py` — creates the class distribution plot
- `prepare_data.py` — data cleaning
- `features.py` — feature extraction
- `train_model.py` — model training and evaluation
- `app.py` — Streamlit app
- `requirements.txt` — required Python packages
- `plots/` — saved evaluation figures
- `models/decision_tree_model.joblib` — trained model

## Run the project

Open a terminal in this project folder and run:

~~~powershell
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
~~~

Open the local address shown by Streamlit. Enter a complete URL beginning with `http://` or `https://`.

To see the EDA, open `eda.ipynb` in VS Code and run its cells.

## Test results

A phishing cutoff of 0.30 was chosen using validation data. On the separate test set, the Decision Tree achieved 86.81% accuracy, compared with 82.57% for always predicting benign.

Phishing precision was 61.2%, recall was 66.6%, F1-score was 0.638, and ROC-AUC was 0.879. The model correctly flagged 11,918 phishing URLs, missed 5,977, and incorrectly flagged 7,564 benign URLs.

Lowering the cutoff improved phishing detection but increased false alarms. For example, the app incorrectly flags `https://www.wikipedia.org` as potential phishing. The model uses URL text only; its predictions do not establish whether a website is safe.
## Dataset setup

The dataset CSV is not included in this GitHub repository. Download the
[Malicious URLs dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)
and place `malicious_phish.csv` inside `candidate_data/`.

The CSV is needed to rerun the EDA and training code. The Streamlit app uses
the included trained model.