# DepBERT: Context-dependent Features Fusion with BERT

This repository contains the Python implementation of the DepBERT model, as described in the paper "Context-dependent Features Fusion with BERT for Evaluating Multi-Turn Customer-Helpdesk Dialogues" by Siu Hin Ng, et al.

## Project Structure

```
depbert/
|-- README.md
|-- requirements.txt
|-- data/
|   |-- sample_data.csv  # Sample data for demonstration
|-- src/
|   |-- __init__.py
|   |-- preprocess.py    # Data preprocessing script
|   |-- model.py         # DepBERT model architecture
|   |-- main.py          # Main script for training and evaluation
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd depbert
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
3.  **Download Spacy model for dependency parsing:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Usage

1.  **Prepare your data:**
    Place your training and testing data in the `data/` directory. The data should be in a CSV format with 'text' and 'label' columns.

2.  **Run the main script:**
    ```bash
    python src/main.py
    ```
