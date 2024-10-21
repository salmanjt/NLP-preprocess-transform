# NLP Preprocessing and Transformation

![NLP-preprocess-transform](images/NLP-preprocess-transform.png)

## Project Description

This project focuses on preprocessing a dataset of published research papers and transforming their textual content into numerical representations suitable for various Natural Language Processing (NLP) applications. The goal is to facilitate advanced NLP tasks such as topic modeling, clustering and classification by converting unstructured text into structured numerical data.

## Key Steps Involved

### Data Acquisition and Preparation

-   **PDF Parsing:** Extracting paper IDs and URLs from an initial PDF file and downloading the full-text research papers for further processing.
-   **Text Aggregation:** Collecting and consolidating the text from each research paper into a structured format for easier handling.

### Information Extraction

-   **Entity Parsing:** Extracting specific entities such as titles, authors, abstracts and the main content from the papers using regular expressions.
-   **Text Preprocessing:** Performing essential NLP preprocessing steps such as sentence segmentation, tokenization, stopword removal (including context-specific stopwords), rare and short token removal and stemming.

### Feature Transformation

-   **Vocabulary Creation:** Building a vocabulary index for each unique token in the corpus.
-   **Count Vector Generation:** Creating sparse count vectors for each paper, representing the frequency of terms.

### Statistical Analysis

-   **Top Terms Identification:** Performing frequency analysis to identify the top 10 most frequent terms in titles and abstracts, as well as the top 10 authors based on the dataset.

Overall, this project demonstrates an effective approach to preprocessing unstructured text data and transforming it into a format that can support advanced text mining and NLP tasks.

## Tools & Technologies

-   Python (re, itertools, concurrent.futures)
-   Cloud Authentication (google.oauth2)
-   PDF Processing (pypdf, pdftotext)
-   NLP (NLTK, PorterStemmer, RegexpTokenizer, MWETokenizer, sent_tokenize)

## Project Tree

```
📦 NLP-preprocess-transform
├─ LICENSE
├─ README.md
├─ requirements.txt
├─ data
│  ├─ input
│  │  ├─ papers.pdf
│  │  ├─ pdf_files
│  │  │  ├─ PP3206.pdf
│  │  │  ├─ PP3234.pdf
│  │  │  ├─ ...
│  │  │  ├─ PP7274.pdf
│  │  │  └─ PP7280.pdf
│  │  └─ stopwords_en.txt
│  ├─ output
│  │  ├─ count_vectors.txt
│  │  ├─ summary_stats.csv
│  │  └─ vocab.txt
│  └─ sample
│     ├─ sample_count_vectors.txt
│     ├─ sample_stats.csv
│     └─ sample_vocab.txt
├─ images
│  └─ NLP-preprocess-transform.png
└─ notebooks
   ├─ 01-preprocess-transform.ipynb
   └─ exports
      └─ 01-preprocess-transform.py
```

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/salmanjt/NLP-preprocess-transform.git
    cd NLP-preprocess-transform
    ```

2. Install required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Data Sources

-   The initial pdf file contains links to academic research papers, stored in a private repository, which are processed using the Google Drive API. The source of the dataset was not disclosed in the project brief.
-   A predefined list of English stopwords is used for text cleaning.

## Future Improvements

-   **Custom Stopword List:** Incorporate domain-specific stopwords to improve the quality of text preprocessing.
-   **Advanced Tokenization:** Explore advanced tokenization techniques like subword tokenization (e.g., Byte-Pair Encoding) or word embeddings.
-   **Visualisation:** Add data visualizations to gain insights into word distributions, topic clusters and trends within the dataset.
-   **Deep Learning:** Implement deep learning techniques like Word2Vec, GloVe, or transformer-based models (e.g., BERT, GPT) for more advanced feature representations.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/salmanjt/NLP-preprocess-transform/blob/main/LICENSE) file for details.
