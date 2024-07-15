# Information Retrieval Project

This project aims to provide hands-on experience with Information Retrieval (IR) tasks, including examining the impact of stop-word removal on vocabulary size, working with the vector space model, and performing classification and clustering tasks.

## Table of Contents
- [Preprocessing Documents](#preprocessing-documents)
- [Text Classification](#text-classification)
- [Text Clustering](#text-clustering)
- [Tools and Libraries](#tools-and-libraries)

## Preprocessing Documents

1. Tokenization: The text was tokenized using NLTK's `word_tokenize` function. The tokens were saved in a file for future reference.

2. Language Modeling: An initial unigram language model was created for the document collection. The model included many stop words, numbers, and punctuation marks. Some query terms like "search," "engine," and "engines" were also present in the list of common words.

3. Linguistic Operations:
   - Non-English word removal: Numbers, punctuation marks, and non-English words were removed from the language model. This significantly reduced the total number of words.
   - Stop-word removal: Stop words were removed using a provided list. This step was performed before case folding to handle both capitalized and lowercase stop words. Removing stop words greatly decreased the total word count but only slightly reduced the number of unique words.
   - Case Folding: All tokens were converted to lowercase. This step did not change the total word count but significantly decreased the count of unique words by mapping capitalized and lowercase variations of the same word to a single term.
   - Stemming: Porter stemmer was used to perform stemming on the tokens. Stemming did not change the total word count but reduced the number of unique words by grouping together words with the same root. However, it resulted in some over-stemming, producing misspelled or meaningless words.
   - Lemmatization: WordNet lemmatizer was used to lemmatize the tokens without stemming. Lemmatization did not change the total word count but slightly decreased the number of unique words by modifying the inflectional forms of words while maintaining their meaning.

4. Language Model Comparison: A new language model was created after each linguistic operation, and the differences were compared and discussed.

| Preprocessing Step          | Total words count | Reduction percentage | Unique words count | Reduction percentage |
|-----------------------------|-------------------|----------------------|--------------------|--------------------|
| No changes                  | 444563            | -                    | 28143              | -                  |
| Keeping English words only  | 308349            | 30%                  | 10315              | 63%                |
| Removing stop words         | 118800            | 61%                  | 9345               | 9%                 |
| Case folding                | 118800            | -                    | 7071               | 24%                |
| Stemming words              | 118800            | -                    | 5284               | 25%                |
| Lemmatization (no stemming) | 118800            | -                    | 7011               | 0.008%             |


The table and image above summarize the impact of each preprocessing step on the language model. 
At the end we decided to keep working on tokens that were lemmatized not stemmed.

## Text Classification

1. Dataset Preparation:
   - Randomly selected 50 documents from different directories.
   - Labeled the documents as relevant or non-relevant to the target category.

2. Preprocessing:
   - Tokenized the documents using NLTK's `word_tokenize` function.
   - Removed stop words using a predefined list of stop words.
   - Performed additional preprocessing steps such as removing rare words that appeared in less than 50% of the documents to handle outliers and improve classification accuracy.

3. Classification:
   - Selected 5 classifiers: Naïve Bayes, SVM, KNN, Rocchio, and Logistic Regression.
   - Performed 10-fold cross-validation, training with 90% of the documents and testing with the remaining 10% using scikit-learn.

4. Results and Analysis:
   - Reported the average accuracy of 10-fold cross-validation for each classifier.
   - KNN achieved the best classification accuracy, benefiting from the data cleaning and outlier handling steps.
   - Examined misclassified documents and explained the reasons for misclassification:
     - False positive case: A non-relevant document was incorrectly classified as relevant due to the frequent occurrence of query terms like "bias" and "search."
     - False negative case: A relevant document was incorrectly classified as non-relevant because the query terms did not appear frequently in the document.

![image](https://github.com/user-attachments/assets/c2e779f3-0fbe-413d-a314-d486ffa9d9b3)

The image above shows the average accuracy of 10-fold cross-validation for each classifier, with KNN achieving the highest accuracy.

## Text Clustering

1. Dataset Preparation:
   - Collected documents from three additional directories based on the provided instructions.
   - Tokenized the documents and created a corpus set.
   - Checked if tokens from each document were present in the corpus set and counted their occurrences.
   - Included a label column "group_name" to indicate the directory each document belongs to.

2. Clustering:
   - Used K-means clustering with scikit-learn to cluster the documents into four clusters based on prior knowledge of the dataset.
   - Evaluated clustering performance using the purity score, which measures the fraction of documents in each cluster that belong to the most common class.

3. Results and Analysis:
   - Initial clustering results:
     - K-means with PCA: Purity score of 64%
     - K-means with t-SNE: Purity score of 70.5%
   - Improved clustering results after data cleaning and outlier removal:
     - K-means with PCA: Purity score of 67.84%
     - K-means with t-SNE: Purity score of 88.94%
   - The four query results were clearly separated into distinct clusters:
     - Green cluster: Search Engine Bias
     - Orange cluster: Gender Bias
     - Red cluster: Algorithmic Transparency
     - Blue cluster: Explainable Artificial Intelligence
   - Some overlap was observed between Explainable Artificial Intelligence and Algorithmic Transparency clusters, but they were still clearly separated.
   - Search Engine Bias and Gender Bias clusters had minimal overlap with other clusters.

![image](https://github.com/user-attachments/assets/78a0f974-a884-40da-86d8-1d5f15da7af3)

The image above shows the improved clustering results using t-SNE, with a purity score of 88.94%. The clusters are well-separated and align with the expected patterns based on the document categories.

Possible errors in K-means clustering include:
- Choosing an inappropriate number of clusters
- Sensitivity to initialization (different random initializations can lead to different results)
- Presence of noise and outliers in the data
- Impact of preprocessing techniques on clustering quality
- High-dimensional and sparse feature space due to term frequency-inverse document frequency (tf-idf) sparsity

Further inspection of the clusters revealed some overlapping between "Algorithmic Transparency" and "Explainable Artificial Intelligence" directories, with mislabeled documents having a notable presence of the same terms with high TF values.

## Tools and Libraries

The project utilized the following tools and libraries:

- Python: The programming language used for implementing the project.
- NLTK (Natural Language Toolkit): Used for text preprocessing tasks such as tokenization, stop-word removal, stemming, and lemmatization.
- scikit-learn: Used for feature extraction, classification, and clustering tasks.
- NumPy: Used for numerical computations and array manipulation.
- Pandas: Used for data manipulation and analysis.
- Matplotlib: Used for data visualization and plotting.

These libraries provided the necessary functionality for text preprocessing, language modeling, classification, clustering, and data visualization throughout the project.
