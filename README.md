# Text Classifier & Sentiment Analysis

A Python-based NLP project that preprocesses customer feedback and classifies sentiment as positive or negative using a neural network.

## Project Overview

This project implements a text preprocessing and classification pipeline with the following components:

### 1. Text Preprocessing
- **Lowercasing**: Convert all text to lowercase
- **Punctuation Removal**: Remove special characters and punctuation
- **Tokenization**: Split text into individual words using NLTK
- **Stopword Removal**: Filter out common English stopwords
- **Lemmatization**: Reduce words to their base form using WordNetLemmatizer

### 2. Feature Extraction
- **Bag of Words (BoW)**: Convert cleaned text into numerical vectors using `CountVectorizer`
- Vector size equals the vocabulary size from the training dataset

### 3. Neural Network Model
- **Input Layer**: Size = vocabulary size (dynamic based on training data)
- **Hidden Layer**: Dense layer with 16 units, ReLU activation
- **Output Layer**: Dense layer with 1 unit, Sigmoid activation (binary classification)
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy, Precision, Recall, F1 Score
- **Training**: 20 epochs, batch size 4

### 4. Dataset
- **Format**: CSV with `feedback` and `label` columns
- **Data**: 50 customer feedback samples
  - Positive feedback: labeled as `1`
  - Negative feedback: labeled as `0`
- **File**: `feedback.csv`

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Setup Virtual Environment

1. **Create virtual environment:**
   ```powershell
   python -m venv .venv
   ```

2. **Activate virtual environment (Windows PowerShell):**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   Or on Command Prompt:
   ```cmd
   .venv\Scripts\activate.bat
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

   Or install manually:
   ```powershell
   pip install numpy pandas scikit-learn nltk tensorflow-cpu
   ```

4. **Download NLTK data (first run only):**
   ```powershell
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

## Usage

### Usage

There are two scripts with different purposes — use the one that matches your workflow:

- **Interactive user mode:** `text_classifier.py` — trains the model and then opens a prompt for live user input (does not print detailed evaluation metrics during interactive use).

```powershell
python text_classifier.py
```

- **Automated test/demo mode:** `test_classifier.py` — trains the model, prints detailed evaluation metrics (Accuracy, Precision, Recall, F1) and runs a few sample predictions non-interactively.

```powershell
python test_classifier.py
```

Interactive mode behavior:
- After training, the script accepts user feedback at the prompt and prints only the sentiment (e.g., `Sentiment: Positive`) so the interaction is clean for end users.

Test/demo mode behavior:
- Prints evaluation metrics and sample predictions; use this when you want model performance information.

## Files

| File | Purpose |
|------|---------|
| `text_classifier.py` | Interactive classifier: trains then accepts user input (no detailed metrics shown) |
| `test_classifier.py` | Test/demo script: trains, prints detailed metrics and sample predictions |
| `feedback.csv` | Training dataset (50 samples) |
| `requirements.txt` | List of Python package dependencies |
| `README.md` | This documentation file |

## Key Features

✅ **Robust Preprocessing**: Comprehensive text cleaning pipeline  
✅ **Neural Network Model**: Deep learning with configurable architecture  
✅ **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score  
✅ **Interactive Interface**: Real-time sentiment prediction  
✅ **Error Handling**: Graceful exception handling throughout  
✅ **Modular Code**: Reusable functions for preprocessing and model creation  

## Dependencies

All required packages are listed in `requirements.txt`:
- `numpy`: Numerical computing
- `pandas`: Data manipulation
- `scikit-learn`: Machine learning utilities
- `nltk`: Natural language processing
- `tensorflow-cpu`: Neural network framework

## Notes

- **TensorFlow CPU**: Uses `tensorflow-cpu` for compatibility without GPU drivers
- **Model Training**: First run trains the model and takes ~1-2 minutes depending on system
- **NLTK Data**: Automatically downloaded on first run via `nltk.download()` calls
- **Batch Size**: Set to 4 for efficient training on small datasets

## Example Output

Test/demo mode (`test_classifier.py`) prints evaluation metrics:

```
Model accuracy: 0.80
Precision: 0.67
Recall: 1.00
F1 Score: 0.80
```

Interactive mode (`text_classifier.py`) shows only user prompts and sentiment results:

```
Enter customer feedback (or type 'exit'): Great product and service!
Sentiment: Positive
```

## Troubleshooting

### Import errors (tensorflow not found)
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Set Python interpreter to `.venv\Scripts\python.exe` in your editor

### NLTK data missing
- Run: `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"`

### Script runs but model accuracy is low
- Add more diverse training samples to `feedback.csv`
- Adjust hyperparameters (epochs, batch_size, hidden layer units) in the script



### project codes done by Yahya Kanjo for Samsung Innovation Campus

