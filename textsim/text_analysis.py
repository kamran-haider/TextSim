"""
A module for performing key phrase extraction and text comparison.
"""
import operator
import numpy as np
from nltk import word_tokenize, sent_tokenize
from nltk import pos_tag_sents
from nltk.chunk.regexp import RegexpParser
from nltk.chunk import tree2conlltags
from nltk.corpus import stopwords
from itertools import chain, groupby
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .utils import text_cleaner


class Text(object):
    """
    Main class to hold documents, represented as strings, and perform various
    operations on it.
    """
    def __init__(self, documents):
        """

        Parameters
        ----------
        asin
        description
        qa_data
        """
        assert len(documents) != 0, "Please provide a list containing at least one string."
        self.documents = documents
        self.vocabulary = None
        self.model = None
        self.tfidf_matrix = None
        self.key_phrases = None


    def build_vocabulary(self):
        """
        Generate a list of candidate phrases from the documents, using POS tagging and chunking
        functionality of nltk.
        """
        stop_words = set(stopwords.words('english'))

        vocabulary = []
        for doc in self.documents:
            words = []
            candidates = []
            clean_doc = text_cleaner(doc)
            sentences = sent_tokenize(clean_doc)
            words.extend([word_tokenize(sentence) for sentence in sentences])
            tagged_words = pos_tag_sents(words)

            grammar = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
            chunker = RegexpParser(grammar)
            # split into a private function
            all_tag = chain.from_iterable([tree2conlltags(chunker.parse(tag)) for tag in tagged_words])
            for key, group in groupby(all_tag, lambda tag: tag[2] != 'O'):
                candidate = ' '.join([word for (word, pos, chunk) in group])
                if key is True and candidate not in stop_words:
                    candidates.append(candidate)
            vocabulary.append(candidates)

        vocabulary = list(chain(*vocabulary))
        vocabulary = list(np.unique(vocabulary))

        self.vocabulary = vocabulary

    def build_tfidf_model(self, min_df=5, max_df=0.8):
        """
        Build a vector space model of the text.

        Parameters
        ----------
        min_df
        max_df

        Returns
        -------
        None

        """
        # Calculate maximum n_gram range
        max_vocab_len = max(map(lambda s: len(s.split(' ')), self.vocabulary))
        # Create tfidf vectorizer based on extracted keyphrases
        tfidf_model = TfidfVectorizer(vocabulary=self.vocabulary, lowercase=True,
                                      ngram_range=(1, max_vocab_len), stop_words=None,
                                      min_df=min_df, max_df=max_df)
        # Create tfidf matrix
        X = tfidf_model.fit_transform(self.documents)
        # Obtain tfidf matrix
        tfidf_matrix = X.toarray()

        self.model = tfidf_model
        self.tfidf_matrix = tfidf_matrix



    def rank_keyphrases(self, num_key=5):
        """
        Use tf-idf weighting to score key phrases

        Parameters
        ----------
        num_key

        Returns
        -------

        """
        sorted_vocabulary = [v[0] for v in sorted(self.model.vocabulary_.items(), key=operator.itemgetter(1))]
        # Sort each row of the tfidf matrix, obtain indices and the flip them left to right
        # This generates a matrix where each row has feature ids ranked by tfidf weights
        sorted_tfidf_index_matrix = np.fliplr(np.argsort(self.tfidf_matrix))
        # return list of top candidate phrase
        key_phrases = list()
        tfidf_weights = list()
        # To generate top phrases in each qa pair, obtain top n indices from
        for doc_index, sorted_feature_indices in enumerate(sorted_tfidf_index_matrix):

            phrases_scores = np.array([(sorted_vocabulary[e], self.tfidf_matrix[doc_index, e])
                                          for e in sorted_feature_indices[0:num_key]])

            phrases = phrases_scores[:, 0]
            weights = phrases_scores[:, 1]

            key_phrases.append(phrases)
            tfidf_weights.append(weights)

        self.key_phrases = np.array(key_phrases)
        self.key_phrase_weights = np.array(tfidf_weights)


    def evaluate_query_document(self, query_text):
        """
        Calculate similarity of a new document against the existing set of documents

        Parameters
        ----------
        query_text

        Returns
        -------

        """

        query_vector = self.model.transform([query_text])
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        qa_coverage_score = np.mean(cosine_similarities)

        return qa_coverage_score, cosine_similarities