![NLP-preprocess-transform](images/NLP-preprocess-transform.png)

# Natural Language Processing (NLP) and Feature Transformation

## Project Description

The primary objective of this project is to preprocess a dataset of published research papers and convert their textual content into numerical representations suitable for various Natural Language Processing (NLP) applications. By transforming unstructured text into structured numerical data, we facilitate advanced NLP tasks such as topic modeling, clustering and classification.

The project involves several key steps:

-   **Data Acquisition and Preparation:** Extracting paper IDs and URLs from an initial PDF file, downloading the research papers and aggregating their textual content.
-   **Information Extraction:** Parsing the text to extract specific entities like titles, authors, abstracts and main content using regular expressions.
-   **Text Preprocessing:** Performing sentence segmentation, tokenization, bigram generation, stopword removal, context-dependent stopword removal, rare token removal, short token removal and stemming.
-   **Feature Transformation:** Creating a vocabulary index and generating sparse count vectors for each paper.
-   **Statistical Analysis:** Identifying the top 10 most frequent terms in titles and abstracts along with the top 10 authors.

Overall, this project demonstrates a practical and comprehensive approach to handling various preprocessing tasks, from tokenization and stemming to feature extraction and statistical analysis. It lays a robust foundation for future analytical endeavors in text mining and natural language processing.

## Project Tree

```
📦 NLP-preprocess-transform
├─ LICENSE
├─ README.md
├─ data
│  ├─ input
│  │  ├─ papers.pdf
│  │  ├─ pdf_files
│  │  │  ├─ PP3206.pdf
│  │  │  ├─ PP3234.pdf
│  │  │  ├─ ...
│  │  │  └─ ...
│  │  └─ stopwords_en.txt
│  ├─ output
│  │  ├─ count_vectors.txt
│  │  ├─ summary_stats.csv
│  │  └─ vocab.txt
│  └─ sample
│     ├─ sample_count_vectors.txt
│     ├─ sample_stats.csv
│     └─ sample_vocab.txt
├─ images
|  ├─ NLP-preprocess-transform.png
└─ notebooks
   └─ 01-preprocess-transform.ipynb
```

## Technologies Used

-   [Python](https://www.python.org/downloads/)
-   [Regex](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_expressions)
-   [Jupyter](https://jupyter.org/)
-   [Google Drive API](https://developers.google.com/drive/api/v3/about-sdk)
-   [NLTK (Natural Language Toolkit)](https://www.nltk.org/)

## Outputs

The project generates several key output files (located in the `data/output` directory):

-   `vocab.txt`: Contains the vocabulary index, mapping each unique token to a numerical index. This is essential for tokenization and creating numerical representations of text.
-   `count_vectors.txt`: Stores the sparse count vectors for each paper. Each line represents a paper and contains pairs of token indices and their corresponding counts.
-   `summary_stats.csv`: Provides a statistical summary, including the top 10 most frequent terms in titles, abstracts and the top 10 most frequent authors.

## Future Improvements

-   **Custom Stopword List:** Incorporate domain-specific stopwords to improve the quality of text preprocessing.
-   **Advanced Tokenization:** Explore advanced tokenization techniques like subword tokenization (e.g., Byte-Pair Encoding) or word embeddings.
-   **Visualisation:** Add data visualizations to gain insights into word distributions, topic clusters and trends within the dataset.
-   **Deep Learning:** Implement deep learning techniques like Word2Vec, GloVe, or transformer-based models (e.g., BERT, GPT) for more advanced feature representations.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/salmanjt/NLP-preprocess-transform/blob/main/LICENSE) file for details.

## Credits

[Project Tree Generator](https://woochanleee.github.io/project-tree-generator)
