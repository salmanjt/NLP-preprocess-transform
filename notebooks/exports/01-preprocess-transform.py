# %% [markdown]
# # Natural Language Preprocessing and Feature Transformation
# 
# **Author:** Salman Tahir  
# **Environment:** Conda 23.7.2, Python 3.10.12
# 

# %% [markdown]
# ---
# 

# %% [markdown]
# **Table of contents**<a id='toc0_'></a>
# 
# -   [Introduction](#toc2_)
# -   [Importing Libraries](#toc3_)
# -   [Reading Data](#toc4_)
# -   [Downloading PDF Files](#toc5_)
# -   [Aggregating Data from PDFs](#toc6_)
# -   [Extracting Information to Entities](#toc7_)
# -   [Preprocessing Data](#toc8_)
#     -   [Sentence Segmentation](#toc8_1_)
#     -   [Tokenization](#toc8_2_)
#     -   [Bigrams](#toc8_3_)
#     -   [Further Token Processing](#toc8_4_)
#     -   [Stemming](#toc8_5_)
# -   [Feature Transformation](#toc9_)
#     -   [Vocab Index](#toc9_1_)
#     -   [Sparse Count Vector](#toc9_2_)
# -   [Statistical Analysis](#toc10_)
# -   [Summary](#toc11_)
# -   [References](#toc12_)
# 
# <!-- vscode-jupyter-toc-config
# 	numbering=false
# 	anchor=true
# 	flat=false
# 	minLevel=1
# 	maxLevel=6
# 	/vscode-jupyter-toc-config -->
# <!-- THIS CELL WILL BE REPLACED ON TOC UPDATE. DO NOT WRITE YOUR TEXT IN THIS CELL -->
# 

# %% [markdown]
# # <a id='toc2_'></a>[Introduction](#toc0_)
# 
# In this project, our primary goal is to bridge the gap between textual information and numerical representations, catering to the needs of advanced Natural Language Processing (NLP) systems and algorithms. Our focus lies in the preprocessing of a diverse dataset of published papers, transforming them into a format that is not only amenable to NLP applications but also highly suitable for downstream modeling tasks.
# 
# Furthermore, we generate a statistical summary of the top 10 most frequent words in the titles, authors and abstracts of the papers.
# 

# %% [markdown]
# # <a id='toc3_'></a>[Importing Libraries](#toc0_)
# 
# The following libraries are used:
# 
# -   `os` (for file path manipulation)
# -   `re` (for regular expressions and pattern matching)
# -   `csv` (for writing to csv files)
# -   `nltk` (for tokenization and stemming using PorterStemmer)
# -   `requests` (for downloading files from the http links)
# -   `concurrent.futures` (for multithreading utilising the ThreadPoolExecutor)
# -   `pdftotext` (for converting pdf tables to text)
# -   `pypdf` (for converting pdf files to text)
# -   `itertools` (for creating combinations of words)
# -   `google.oauth2` (for authenticating with google drive API)
# -   `collections` (for counting the frequency of words)
# 

# %%
import os
import re
import csv
import nltk
import requests
import pdftotext
import itertools
from pypdf import PdfReader
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from nltk.collocations import *
from nltk import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import MWETokenizer


# %% [markdown]
# # <a id='toc4_'></a>[Reading Data](#toc0_)
# 
# We start by reading the initial PDF file and extracting the links to the papers in the PDF table.
# 
# -   The extracted data is stored in a dictionary.
# -   The keys of the dictionary are the paper IDs.
# -   The values of the dictionary are the links to the papers.
# 

# %%
# Open the PDF file and read its contents
with open("../data/input/papers.pdf", "rb") as file:
    reader = PdfReader(file)
    num_pages = len(reader.pages)
    data = {}

    # Extract URLs and filenames from table in the PDF file
    for page in range(num_pages):
        page_obj = reader.pages[page]
        page_text = page_obj.extract_text()
        lines = page_text.split("\n")
        for line in lines:
            if "http" in line:
                # Extract filename from the line before URL
                filename = lines[lines.index(line) - 1]
                data[filename] = line

    # Create a new dictionary with modified keys
    articles = {key[:-4]: value for key, value in data.items()}


# %%
# Print first item in the dictionary
for key, value in list(articles.items())[:1]:
    print(f"Filename: {key}\nURL: {value}\n")


# %% [markdown]
# # <a id='toc5_'></a>[Downloading PDF Files](#toc0_)
# 
# -   We start by setting up our credentials for the google drive API.
#     -   Doing so allows us to bypass the rate limit enforced by google drive.
# -   Download the PDF files into a directory called `pdf_files`.
#     -   Note, that we also ensure to append the file extension to the file name.
# -   Using multithreading we download the files in parallel.
# 
# **Before running the code block below, please ensure you have the `credentials.json` file available in the same directory as this notebook.**
# 

# %%
# # Set up credentials for Google Drive API
# credentials = service_account.Credentials.from_service_account_file(
#     'credentials.json')

# # Create a directory to store PDF files
# if not os.path.exists("../data/input/pdf_files/"):
#     os.makedirs("../data/input/pdf_files/")


# def download_pdf(url, filename):
#     """
#     Downloads the PDF file from the given URL and saves it to the pdf_files directory.
#     :param url: URL of the PDF file
#     :param filename: Filename of the PDF file
#     """
#     try:
#         # Use credentials to access the Google Drive API
#         response = requests.get(
#             url, headers={"Authorization": f"Bearer {credentials.token}"})
#         with open(f"pdf_files/{filename}.pdf", "wb") as f:
#             f.write(response.content)
#         # Print filename of the downloaded PDF file to verify status
#         print(f"Downloaded {filename}.pdf")
#     except:
#         # Print filename of the PDF file that failed to download
#         print(f"Error downloading {filename}.pdf from URL: {url}")


# # To download PDF files in parallel
# with ThreadPoolExecutor() as executor:
#     for filename, url in articles.items():
#         executor.submit(download_pdf, url, filename)


# %%
# Count the number of files in directory
num_files = len([f for f in os.listdir('../data/input/pdf_files/')
                if os.path.isfile(os.path.join('../data/input/pdf_files/', f))])

print(f"There are {num_files} files in the {'pdf_files'} directory.")


# %% [markdown]
# # <a id='toc6_'></a>[Aggregating Data from PDFs](#toc0_)
# 
# Now, we extract all text from the PDF files into a dictionary.
# 
# -   We remove the file extension from the file name and use it as the key for the dictionary.
# -   The values of the dictionary are the extracted text from the PDF files.
# 

# %%
# Set path to the directory containing the PDF files
pdf_directory = "../data/input/pdf_files/"
pdf_files = os.listdir(pdf_directory)

# Create a dictionary to store text from PDF files
text_dict = {}

# Iterate through PDF files in the directory and extract text
for file_name in pdf_files:
    # Check to confirm that the file is a PDF file
    if file_name.endswith(".pdf"):
        file_path = os.path.join(pdf_directory, file_name)
        with open(file_path, "rb") as pdf_file:
            pdf = pdftotext.PDF(pdf_file)
            text = "\n".join(pdf)
            # Remove file extension from the filename
            key = file_name[:-4]
            # Add text to our dictionary
            text_dict[key] = text


# %% [markdown]
# # <a id='toc7_'></a>[Extracting Information to Entities](#toc0_)
# 
# Once we have have our dictionary containing the extracted text from the PDF files, we can extract the required entities.
# 
# We start by compiling the regular expressions for extracting the required entities.
# 
# **Regular expression for extracting Titles**
# 
# ```python
# r'^(.+?)Authored'
# ```
# 
# -   `^(.+?)` matches all text until the word `Authored` is found.
# -   `Authored` matches the literal word `Authored`.
# 
# **Regular expression for extracting Authors**
# 
# ```python
# r'(?<=Authored by:)(?:\s*)([A-Za-z\s.?-]+)(?=\n\s*Abstract)'
# ```
# 
# -   `(?<=Authored by:)` matches the literal word `Authored by:` and ensures that the match is not included in the result.
# -   `(?:\s*)` matches any whitespace characters.
# -   `([A-Za-z\s.?-]+)` matches all text until the next newline character is found.
# -   `(?=\n\s*Abstract)` matches the literal word `Abstract` and ensures that the match is not included in the result.
# 
# **Regular expression for extracting Abstract**
# 
# ```python
# r'Abstract(.+?)\s*1\s*[\n\s]*Paper Body'
# ```
# 
# -   `Abstract` matches the literal word `Abstract`.
# -   `(.+?)` matches all text until the word `Paper Body` is found.
# -   `\s*1\s*` matches the literal word `1`.
# -   `[\n\s]*` matches any whitespace characters.
# -   `Paper Body` matches the literal word `Paper Body`.
# 
# **Regular expression for extracting Paper Bodies**
# 
# ```python
# r'1\s*Paper Body(.+?)2\s*References'
# ```
# 
# -   `1\s*Paper Body` matches the literal word `1 Paper Body`.
# -   `(.+?)` matches all text until the word `2 References` is found.
# -   `2\s*References` matches the literal word `2 References`.
# 
# Once, we have used the regular expressions to extract the required entities, we can perform some preprocessing on the data.
# 
# -   We remove newlines and multiple spaces from the data.
# -   Note that before removing multiple spaces for authors, we perform a split using double space as the delimiter.
# -   Finally, the data is stored in a dictionary.
#     -   The keys of the dictionary are the paper IDs.
#     -   The values of the dictionary are the extracted entities.
# 

# %%
# Compile regular expression for titles
TITLE_PATTERN = re.compile(r'^(.+?)Authored', re.DOTALL)

# Compile regular expression for authors
AUTHOR_PATTERN = re.compile(
    r'(?<=Authored by:)(?:\s*)([A-Za-z\s.?-]+)(?=\n\s*Abstract)', re.DOTALL)

# Compile regular expression for abstracts
ABSTRACT_PATTERN = re.compile(
    r'Abstract(.+?)\s*1\s*[\n\s]*Paper Body', re.DOTALL)

# Compile regular expression for paper bodies
PAPER_PATTERN = re.compile(
    r'1\s*Paper Body(.+?)2\s*References', re.DOTALL)


# Create dictionaries to store extracted data
titles = {}
authors = {}
abstracts = {}
papers = {}


for file_name, text in text_dict.items():
    # Extract title
    title = TITLE_PATTERN.findall(text)
    # Remove extra spaces from title
    title = re.sub(r'\s+', ' ', title[0].strip())
    # Add title to dictionary
    titles[file_name] = title

    # Extract author
    author = AUTHOR_PATTERN.findall(text)
    authors_list = []
    for a in author:
        # Remove newlines and extra spaces from author name
        clean_author = re.sub(r'[\n\r]+', ' ', a.strip())
        # Split multiple authors using delimiter "  "
        authors_list.extend(clean_author.split("  "))
        # Remove empty strings from list
        authors_list = list(filter(None, authors_list))
        # Add author to dictionary
        authors[file_name] = authors_list

    # Extract abstract
    abstract = ABSTRACT_PATTERN.findall(text)
    # Remove newlines and extra spaces from abstract
    clean_abstract = re.sub(r'\s+', ' ', abstract[0].strip())
    clean_abstract = re.sub(r'\r\n', ' ', clean_abstract)
    # Add the abstract to dictionary
    abstracts[file_name] = clean_abstract

    # Extract the paper
    paper = PAPER_PATTERN.findall(text)
    # Remove newlines and extra spaces from the paper
    clean_paper = re.sub(r'\s+', ' ', paper[0].strip())
    # Add the paper to our dictionary
    papers[file_name] = clean_paper


# %%
# Print first paper title
for key, value in list(titles.items())[:1]:
    print(f"Filename: {key}\nTitle: {value}\n")

# Print first paper authors
for key, value in list(authors.items())[:1]:
    print(f"Filename: {key}\nAuthors: {value}\n")

# Print first paper abstract
for key, value in list(abstracts.items())[:1]:
    print(f"Filename: {key}\nAbstract: {value}\n")

# Print first paper body
for key, value in list(papers.items())[:1]:
    print(f"Filename: {key}\nPaperBody: {value}\n")


# %% [markdown]
# # <a id='toc8_'></a>[Preprocessing Data](#toc0_)
# 

# %% [markdown]
# ## <a id='toc8_1_'></a>[Sentence Segmentation](#toc0_)
# 
# -   By iterating over each key value pair in the papers dictionary we:
#     -   Perform sentence tokenization using the `sent_tokenize` function from the `nltk` library.
#     -   We account for the case that we require capitalisation to stay intact.
#         -   Doing so by using `isupper()`
#     -   We then append each normalised word to the list of normalised sentences.
#     -   Finally, we update the value in the `papers` dictionary with the normalised sentences.
# 

# %%
# Iterate through papers dictionary and perform sentence segmentation
for file_name, paper in papers.items():
    sentences = sent_tokenize(paper)
    # Create a list to store normalized sentences
    normalized_sentences = []
    for sentence in sentences:
        # Split the sentence into tokens
        words = sentence.split()
        for i, word in enumerate(words):
            if i == 0 or i == len(words) - 1 or word.isupper():
                # Keep capital tokens at the beginning, end or standalone
                normalized_word = word
            else:
                # Normalize lowercase for non initial capital tokens
                normalized_word = word.lower()
            # Append normalized word to list
            normalized_sentences.append(normalized_word)
    # Update the papers dict with our normalized sentences
    papers[file_name] = ' '.join(normalized_sentences)


# %% [markdown]
# We also remove any numbers/digits present in the sentences as they are not required for the analysis.
# 

# %%
# Remove unnecessary digits from papers
papers = {key: re.sub(r'\d+', '', value) for key, value in papers.items()}


# %% [markdown]
# ## <a id='toc8_2_'></a>[Tokenization](#toc0_)
# 
# Using the regular expression provided for in the specification, we perform word tokenization on the sentences.
# 

# %%
# Define regular expression pattern for words
WORD_PATTERN = r"[A-Za-z]\w+(?:[-'?]\w+)?"

# Create tokenizer object
tokenizer = RegexpTokenizer(WORD_PATTERN)

# Iterate through the papers dictionary and tokenize words
for file_name, paper in papers.items():
    words = tokenizer.tokenize(paper)
    # Update papers dict with tokenized words
    papers[file_name] = words


# %% [markdown]
# ## <a id='toc8_3_'></a>[Bigrams](#toc0_)
# 
# In this step we generate bigrams from the tokens using the `nltk` library.
# 
# -   Using list comprehension we remove bigrams that contain any stopwords or have the separator `"__"`
# -   We store the top 200 bigrams in filtered_bigrams.
# 
# Finally, we retokenize the words in all papers using the multi word expression tokenizer, and update the value in the papers dictionary with the new tokenized words.
# 

# %%
# Load stopwords file
with open("../data/input/stopwords_en.txt") as f:
    stopwords = set(f.read().splitlines())


# Create a single list of all words in papers
all_words = [word for words in papers.values() for word in words]


# Create frequency distribution of all words
bigram_freq = nltk.FreqDist(nltk.bigrams(all_words))


# Remove bigrams that contain stopwords or '__'
filtered_bigrams = [(w1, w2) for (
    w1, w2) in bigram_freq if w1 not in stopwords and w2 not in stopwords]
filtered_bigrams = [(w1, w2) for (
    w1, w2) in filtered_bigrams if "__" not in w1 and "__" not in w2]


# Store the top 200 bigrams
top_200_bigrams = filtered_bigrams[:200]


# Create a multi word expression tokenizer with the top 200 bigrams
mwetokenizer = MWETokenizer(top_200_bigrams, separator='__')


# Retokenize the words in all papers using the multi word expression tokenizer
for file_name, words in papers.items():
    papers[file_name] = mwetokenizer.tokenize(words)


# %% [markdown]
# ## <a id='toc8_4_'></a>[Further Token Processing](#toc0_)
# 
# We perform the following preprocessing steps on the tokens based on the given specification.
# 

# %% [markdown]
# -   Using list comprehension we remove any stopwords.
# 

# %%
# Iterate through papers dictionary and remove stopwords
for file_name, words in papers.items():
    papers[file_name] = [word for word in words if word not in stopwords]


# %% [markdown]
# -   By computing the frequency distribution of the tokens in all papers, we identify the context-dependent stopwords (that appear in 95% of the papers) and remove them.
# 

# %%
# Create a frequency distribution of all words
token_freq = nltk.FreqDist(
    [word for words in papers.values() for word in words])

# Create a set of context-dependent stopwords that appear in more than 95% of papers
context_dependent_stopwords = set(
    [token for token, freq in token_freq.items() if freq/len(papers) >= 0.95])

# Iterate through papers dictionary and remove context-dependent stopwords
for file_name, words in papers.items():
    papers[file_name] = [
        word for word in words if word not in context_dependent_stopwords]


# %% [markdown]
# -   We identify rare tokens (that appear in less than 3% papers) and remove them.
# 

# %%
# Create a set of rare tokens that appear in less than 3% of papers
rare_tokens = set(
    [token for token, freq in token_freq.items() if freq/len(papers) < 0.03])

# Iterate through papers dictionary and remove these rare tokens
for file_name, words in papers.items():
    papers[file_name] = [word for word in words if word not in rare_tokens]


# %% [markdown]
# -   Finally, we remove characters/symbols that are less than 3 characters long.
# 

# %%
# Iterate through papers dictionary and remove words with length less than 3
for file_name, words in papers.items():
    papers[file_name] = [word for word in words if len(word) >= 3]


# %% [markdown]
# ## <a id='toc8_5_'></a>[Stemming](#toc0_)
# 
# Once we have preprocessed the tokens according to the specification, we perform stemming using the PorterStemmer from the `nltk` library.
# 
# Additionally, we account for cases where the original token was in uppercase or capitalized by preserving the original capitalization in the stemmed version of the token (this was required in the specification).
# 

# %%
# Create stemmer object
stemmer = PorterStemmer()

# Iterate through papers dictionary and stem words
for file_name, words in papers.items():
    # Create initial list to store stemmed words
    stemmed_words = []
    for word in words:
        # Only stem words with length greater than 3
        if len(word) > 3:
            # Check if word is uppercase, lowercase or TitleCase (we need to preserve this)
            if word.isupper():
                stemmed_words.append(stemmer.stem(word).upper())
            elif word[0].isupper():
                stemmed_words.append(stemmer.stem(word).capitalize())
            else:
                stemmed_words.append(stemmer.stem(word))
    # Update papers dict with stemmed words
    papers[file_name] = stemmed_words


# %% [markdown]
# # <a id='toc9_'></a>[Feature Transformation](#toc0_)
# 
# Iterating over all values in the `papers` dictionary we aggregate the tokens into a list.
# 
# We remove all duplicates from the list and sort the list in alphabetical order.
# 

# %%
# Create a single list of all words in papers
all_tokens = []
for word in papers.values():
    all_tokens.extend(word)

# Create a set of all tokens
all_tokens = set(all_tokens)

# Sort the tokens alphabetically
all_tokens = sorted(all_tokens)


# %%
# Print the number of tokens
print(f"Number of tokens: {len(all_tokens)}")


# %% [markdown]
# We map each token in our list of tokens to the index in the list (that was alphabetically sorted).
# 
# Doing so, we can create a mapping of each token to its index
# 

# %%
# Create a dict to store index of each token
token_index = {}
for i, token in enumerate(all_tokens):
    token_index[token] = i


# %% [markdown]
# ## <a id='toc9_1_'></a>[Vocab Index](#toc0_)
# 

# %% [markdown]
# Now, we output the the vocabulary index file with the format given in the specification.
# 

# %%
# Write vocab index to file
with open("../data/output/vocab.txt", "w") as f:
    for token, index in token_index.items():
        f.write(f"{token}:{index}\n")


# %% [markdown]
# ## <a id='toc9_2_'></a>[Sparse Count Vector](#toc0_)
# 

# %% [markdown]
# Now, we iterate over each paper in our `papers` dictionary and create a sparse count vector for the paper using the token index dictionary.
# 
# The format is kept same as the specification where we have the paper ID (filename) and sparse count vector for each paper.
# 

# %%
# Write count vectors to file
with open("../data/output/count_vectors.txt", "w") as f:
    for paper_id, paper in papers.items():
        # Create dict to store counts of each token
        counts = {}
        for token in paper:
            if token in counts:
                counts[token] += 1
            else:
                counts[token] = 1
        # Create list to store sparse counts
        sparse_counts = []
        for token, count in counts.items():
            if token in token_index:
                # Get index of token from token_index dict
                index = token_index[token]
                sparse_counts.append(f"{index}:{count}")
        # Join sparse counts with commas
        sparse_counts_str = ",".join(sparse_counts)
        f.write(f"{paper_id},{sparse_counts_str}\n")


# %% [markdown]
# # <a id='toc10_'></a>[Statistical Analysis](#toc0_)
# 
# We previously extracted all required data from the PDF files for the entities required for analysis.
# 
# Now, we perform some statistical analysis on the data to find the 10 most frequent words in the titles, authors and abstracts.
# 

# %%
# Load stopwords file again (optional, we also have it in memory)
with open("../data/input/stopwords_en.txt") as f:
    stopwords = set(f.read().splitlines())


# %% [markdown]
# Set variables for pattern matching as per the specification.
# 

# %%
# Set regular expression pattern for words
WORD_PATTERN = r"[A-Za-z]\w+(?:[-'?]\w+)?"

# Compile regular expression
PATTERN = re.compile(WORD_PATTERN)


# %% [markdown]
# Here, we perform the following steps:
# 
# -   Tokenize the titles and abstracts using the regular expression provided in the specification.
#     -   It was set as a global variable in the code block above.
#     -   The authors did not require tokenization.
# -   We then identify the top 10 most frequent words in the titles, authors and abstracts.
# 
# Note that for the entities we had already created dictionaries at the start of the notebook.
# 

# %%
# Create empty lists for titles and abstracts
all_titles = []
all_abstracts = []

# Iterate through titles, tokenize and remove stopwords
for paper_id, title in titles.items():
    # Tokenize the title and remove stopwords
    title_tokens = PATTERN.findall(title.lower())
    title_tokens = [token for token in title_tokens if token not in stopwords]
    all_titles.extend(title_tokens)

# Count the most frequent terms in the titles and abstracts
titles_count = Counter(all_titles)
top10_titles = [term for term, count in sorted(
    titles_count.items(), key=lambda x: (-x[1], x[0]))[:10]]


# Iterate through abstracts, tokenize and remove stopwords
for paper_id, abstract in abstracts.items():
    # Tokenize the abstract and remove stopwords
    abstract_tokens = PATTERN.findall(abstract.lower())
    abstract_tokens = [
        token for token in abstract_tokens if token not in stopwords]
    all_abstracts.extend(abstract_tokens)

# Count the most frequent terms in the titles and abstracts
abstracts_count = Counter(all_abstracts)
top10_abstracts = [term for term, count in sorted(
    abstracts_count.items(), key=lambda x: (-x[1], x[0]))[:10]]


# Using list comprehension, create a list of all authors
authors_flat = [author for authors_list in authors.values()
                for author in authors_list]

# Remove whitespace from author names
authors_flat = [author.strip() for author in authors_flat]

# Count the most frequent authors
authors_counter = Counter(authors_flat)
top10_authors = [author for author, count in sorted(
    authors_counter.items(), key=lambda x: (-x[1], x[0]))[:10]]


# %% [markdown]
# Finally, we output our statistical summary to a CSV file using the `csv` library.
# 

# %%
# Write stats to file
with open('../data/output/summary_stats.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow([
        'top10_terms_in_abstracts',
        'top10_terms_in_titles',
        'top10_authors'])
    # Write rows
    writer.writerows(zip(
        top10_abstracts,
        top10_titles,
        top10_authors))


# %% [markdown]
# # <a id='toc11_'></a>[Summary](#toc0_)
# 
# In this project, we have systematically transformed a collection of research papers into a structured numerical format suitable for advanced Natural Language Processing (NLP) applications and downstream modelling tasks. The process encompassed several key steps, each crucial for preparing the textual data for effective analysis.
# 
# ### Data Acquisition and Preparation
# 
# -   **Reading Data:** We began by reading an initial PDF file containing a table of contents with links to various research papers. By parsing this PDF, we extracted the paper IDs and their corresponding URLs, storing them in a dictionary for easy access.
# 
# -   **Downloading PDF Files:** Utilising multithreading for efficiency, we downloaded all the listed research papers in PDF format. This step involved setting up authentication credentials to interact with the Google Drive API, ensuring seamless and authorised access to the files.
# 
# -   **Aggregating Data from PDFs:** After downloading, we extracted the textual content from each PDF file. This text was stored in a dictionary, with the paper IDs serving as keys and the extracted text as values.
# 
# ### Information Extraction
# 
# -   Using regular expressions, we parsed the textual data to extract specific entities,
#     -   Titles: Captured by identifying text preceding the word "Authored".
#     -   Authors: Extracted by locating the text following "Authored by:" and preceding "Abstract".
#     -   Abstracts: Identified by isolating text between "Abstract" and "Paper Body".
#     -   Paper Bodies: Extracted by capturing the text within the main content of the paper, excluding references
# 
# ### Text Preprocessing
# 
# -   **Sentence Segmentation:** We tokenized the paper bodies into sentences using NLTK's sent_tokenize function. During this process, we normalised the text by handling capitalisation appropriately and removing extraneous digits.
# 
# -   **Tokenization:** Each sentence was further broken down into words using a regular expression pattern designed to capture meaningful tokens while excluding punctuation and numbers.
# 
# -   **Bigrams Generation:** We generated bigrams (pairs of consecutive words) from the tokens to capture contextual relationships. After filtering out bigrams containing stopwords or undesired characters, we identified the top 200 most frequent bigrams and integrated them back into our tokenized data using a multi-word expression tokenizer.
# 
# ### Further Token Processing
# 
# -   **Stopword Removal:** Common stopwords were removed using a predefined list, reducing noise and focusing on significant words.
# 
# -   **Context-Dependent Stopword Removal:** Tokens appearing in 95% or more of the papers were considered context-dependent stopwords and were removed to enhance the specificity of our data.
# 
# -   **Rare Token Removal:** Tokens present in less than 3% of the papers were deemed too infrequent to contribute meaningfully to the analysis and were excluded.
# 
# -   **Short Token Removal:** Tokens shorter than three characters were removed to eliminate insignificant words and abbreviations.
# 
# -   **Stemming:** We applied the Porter Stemmer algorithm to reduce words to their root forms, which helps in grouping similar words and reducing dimensionality. We ensured that original capitalisation was preserved in the stemmed tokens.
# 
# ### Feature Transformation
# 
# -   **Vocabulary Index Creation:** We compiled a sorted list of all unique tokens across the papers to create a vocabulary index. Each token was mapped to a unique index, facilitating efficient lookup and vectorization.
# 
# -   **Sparse Count Vector Generation:** For each paper, we constructed a sparse count vector that represents the frequency of each token from the vocabulary within the paper. This vectorization is crucial for quantitative analysis and machine learning models.
# 
# ### Statistical Analysis
# 
# -   We performed a statistical analysis to determine the top 10 most frequent terms in the titles, abstracts and authors:
#     -   Titles: Analysed to find prevalent topics and themes.
#     -   Abstracts: Examined to uncover common research focuses and terminologies.
#     -   Authors: Identified the most prolific contributors within the dataset.
# 
# ### Output Generation
# 
# -   **Vocabulary Index File (`vocab.txt`):** Contains the mapping of each token to its unique index, serving as a reference for tokenization and vectorization.
# 
# -   **Count Vectors File (`count_vectors.txt`):** Stores the sparse count vectors for each paper, essential for downstream modelling and analysis.
# 
# -   **Statistical Summary File (`summary_stats.csv`):** Provides the lists of the top 10 terms in titles and abstracts, and the top 10 authors, offering immediate insights into the dataset's content.
# 
# Through this comprehensive process, we have effectively bridged the gap between unstructured textual data and structured numerical representations. The transformed data is now well-prepared for advanced NLP applications, including topic modelling, clustering and classification tasks. This project not only facilitates deeper insights into the research papers but also establishes a robust foundation for future analytical endeavours in text mining and natural language processing.
# 

# %% [markdown]
# # <a id='toc12_'></a>[References](#toc0_)
# 
# [1] [Information on using NLTK library for text preprocessing](https://www.nltk.org/book/)
# 
# [2] [Using ThreadPool for parallel processing](https://www.digitalocean.com/community/tutorials/how-to-use-threadpoolexecutor-in-python-3#step-2-using-threadpoolexecutor-to-execute-a-function-in-threads)
# 
# [3] [Sorting data with lambda function](https://blogboard.io/blog/knowledge/python-sorted-lambda/)
# 
# [4] [Compiling and testing regex patterns](https://pythex.org/)
# 


