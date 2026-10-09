# Breast Cancer Classification Using an Artificial Neural Network

## Project Overview

This project develops and evaluates an artificial neural network (ANN) using TensorFlow/Keras to classify breast tumours as **benign** or **malignant** using the Breast Cancer Wisconsin (Diagnostic) dataset.

The project demonstrates a supervised machine-learning workflow, including data preparation, feature standardisation, model training, evaluation, consideration of overfitting, and reflection on ethical issues such as bias, fairness, and the limitations of AI in healthcare.

> **Important:** This is an educational project, not a clinically validated diagnostic system. Its predictions must not be used as a substitute for professional medical advice, diagnosis, or treatment.

## Objectives

- Prepare the dataset for binary classification.
- Separate predictor features from the diagnosis target.
- Encode diagnosis labels consistently.
- Split data into training and test sets while preserving class proportions.
- Standardise numerical features without leaking information from the test set.
- Build and train an ANN using TensorFlow/Keras.
- Evaluate performance with accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.
- Examine possible overfitting and consider improvement strategies.
- Discuss limitations, bias, fairness, and responsible use in healthcare.

## Dataset

**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Repository:** UCI Machine Learning Repository  
**Dataset page:** https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic  
**Dataset DOI:** https://doi.org/10.24432/C5DW2B

The dataset contains measurements computed from digitised images of fine-needle aspirate samples of breast masses. The target variable identifies a diagnosis as benign or malignant.

### Target encoding

| Diagnosis | Encoded value |
|---|---:|
| Benign (B) | 0 |
| Malignant (M) | 1 |

The project workflow described in the report used 569 observations and 30 numerical predictor features after excluding the identifier and any non-feature column such as `Unnamed: 32`. Confirm the exact row and column counts against the version of the dataset used in your notebook.

## Methodology

### 1. Data preparation

- Inspect the dataset structure and missing values.
- Remove identifier columns and empty or non-predictive columns where appropriate.
- Separate the numerical predictor features (`X`) from the diagnosis target (`y`).
- Encode benign as `0` and malignant as `1`.
- Use a stratified 80/20 train-test split with `random_state=42`.
- Fit `StandardScaler` on the training features only, then transform both training and test features.

### 2. ANN architecture

The baseline model used a feed-forward neural network with the following structure:

| Layer | Configuration |
|---|---|
| Input | 30 numerical features |
| Hidden layer 1 | 16 neurons, ReLU activation |
| Hidden layer 2 | 8 neurons, ReLU activation |
| Output | 1 neuron, sigmoid activation |

The baseline training configuration included the Adam optimiser, binary cross-entropy loss, accuracy tracking, a batch size of 32, and training for up to 100 epochs. Early stopping and learning-rate reduction callbacks were used to help manage training and validation behaviour. Check the notebook for the exact callback settings used in the final run.

### 3. Evaluation

The model was assessed using:
- Accuracy
- Precision
- Recall (sensitivity)
- F1-score
- ROC-AUC
- Confusion matrix and class-level classification report

The test set should be kept separate from model selection and hyperparameter tuning. Use validation data or cross-validation when comparing alternative configurations.

## Baseline Results

The recorded baseline evaluation was:

| Metric | Result |
|---|---:|
| Test accuracy | 98.25% |
| Malignant-class precision | 97.62% |
| Malignant-class recall | 97.62% |
| Malignant-class F1-score | 97.62% |
| ROC-AUC | 0.9983 |

The test set contained 114 observations. Based on the reported metrics and class supports, the model correctly classified approximately 112 of 114 observations. The inferred confusion matrix contains one false positive and one false negative; verify this against the actual confusion matrix printed by the notebook before relying on it.

### Interpretation

The baseline showed strong discrimination on this particular test set. However, the test set is relatively small, and high test performance does not demonstrate clinical validity, fairness across demographic groups, reliable probability calibration, or generalisation to other hospitals and populations.

Training accuracy approached 99.45%, while validation accuracy remained around 96.70% and validation loss increased slightly during continued training. This pattern suggests possible overfitting. Dropout regularisation was identified as a possible experiment, but no improvement should be claimed unless the modified model is trained and evaluated and the results are documented.

## Suggested Project Structure

Your actual files may be named differently; update this outline to match the repository.

```text
project/
├── README.md
├── requirements.txt
├── notebooks/
│   └── breast_cancer_ann.ipynb
├── data/
│   └── breast_cancer_wisconsin.csv
└── outputs/
    ├── evaluation_metrics.txt
    ├── confusion_matrix.png
    └── training_history.png
```

Do not commit private data, credentials, virtual environments, or large generated files unnecessarily. If the dataset is downloaded separately, follow the repository's terms and cite the dataset source.

## Installation

Use Python 3.9 or another Python version compatible with the TensorFlow version installed in your environment. TensorFlow compatibility varies by operating system and version, so check the official installation guide if installation fails.

Create and activate a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the main dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install tensorflow pandas numpy scikit-learn matplotlib seaborn jupyter
```

If your notebook uses additional packages, include them in `requirements.txt`. For reproducibility, record the versions that worked in your environment, for example with:

```bash
python -m pip freeze > requirements.txt
```

## Running the Project

1. Download the dataset from the UCI Machine Learning Repository.
2. Place the CSV in the location expected by the notebook, or update the notebook's data path.
3. Activate the environment and install the dependencies.
4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open the project notebook and run the cells in order.
6. Confirm the preprocessing, training, evaluation metrics, and plots are generated.
7. Record the final model configuration and results in the report.

## Responsible AI, Bias, and Fairness

Potential limitations include sampling bias, underrepresentation of relevant populations, measurement differences, and changes between the dataset and a future clinical setting. These are risks to investigate; the overall evaluation metrics in this project do **not** establish that the model is biased or fair.

Further work should consider:
- External validation on independent datasets.
- Subgroup evaluation when suitable demographic or clinical data are available and appropriate to use.
- Sensitivity, specificity, false-negative and false-positive rates.
- Confidence intervals and sample sizes.
- Probability calibration and classification-threshold selection.
- Clinical oversight and relevant ethical and regulatory review.

A false negative—classifying a malignant case as benign—could delay further investigation. Therefore, the model must not be used as a stand-alone diagnostic tool.

## AI Usage Disclosure

OpenAI ChatGPT was used as a supporting learning and development tool for explanations of machine-learning concepts, coding guidance, troubleshooting, interpretation of evaluation metrics, and organisation of project documentation. AI suggestions were treated as guidance and should be checked against executed code, notebook outputs, and reliable technical sources. Any description of AI use should reflect the assistance actually received and the verification actually performed.

## References

- National Institute of Standards and Technology. (2023). *Artificial intelligence risk management framework (AI RMF 1.0)* (NIST AI 100-1). https://doi.org/10.6028/NIST.AI.100-1
- Scikit-learn developers. (n.d.). *Model evaluation: Quantifying the quality of predictions*. https://scikit-learn.org/stable/modules/model_evaluation.html
- TensorFlow. (n.d.). *tf.keras.layers.Dropout*. https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
- UCI Machine Learning Repository. (n.d.). *Breast Cancer Wisconsin (Diagnostic)*. https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic
- World Health Organization. (2021). *Ethics and governance of artificial intelligence for health: WHO guidance*. https://www.who.int/publications/i/item/9789240029200

## License and Disclaimer

Add a license if you intend to publish or share the code. Dataset access and reuse are subject to the dataset repository's stated terms.

This project is for educational and research purposes only. It has not been clinically validated and must not be used to diagnose, rule out, or treat cancer.
