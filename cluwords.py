import timeit
import warnings

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from alfa_algorithms import AlfaKnn


class Cluwords:
    """
    Description
    -----------
    Create the cluwords DataFrame from the pre-treined embedding model (e.g., GloVe, Wiki News - FastText).

    Parameters
    ----------
    algorithm: str
        The algorithm to use as cluwords distance limitation (alfa).
        'knn' : use NearestNeighbors.
        'k-means' : use K-Means.
        'dbscan' : use DBSCAN.
    embedding_file_path: str
        The path to embedding pre-treined model.
    n_words: int
        Number of words in the dataset.
    k_neighbors: boolean
        Number of neighbors desire for each cluword.
    cosine_lim: float, (default = .85)
        The cosine limit to consider the value of cosine siliarity between two words in the model.

        Note: if two words have the cosine similiarity under cosine limit, the value of cosine similiarty
            is equal zero.
    n_jobs: int, (default = 1)
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
        Affects only :meth:`kneighbors` and :meth:`kneighbors_graph` methods.
    verbose: int, (default = 0)
        Enable verbose output.

    Attributes
    ----------

    """

    def __init__(self, dataset, algorithm, embedding_file_path, n_words, k_neighbors, threshold=.85, n_jobs=1, verbose=0):
        if verbose:
            print('K: {}'.format(k_neighbors))
            print('Cossine: {}'.format(threshold))

        if algorithm == 'knn_cosine':
            print('kNN...')
            knn = AlfaKnn(threshold=threshold,
                          n_threads=n_jobs)
            knn.create_cosine_cluwords(input_vector_file=embedding_file_path,
                                       n_words=n_words,
                                       k_neighbors=k_neighbors,
                                       dataset=dataset)
        elif algorithm == 'knn_mahalanobis':
            print('kNN Mahalanobis...')
            knn = AlfaKnn(threshold=threshold,
                          n_threads=n_jobs)
            knn.create_mahalanobis_cluwords(input_vector_file=embedding_file_path,
                                            n_words=n_words,
                                            k_neighbors=k_neighbors,
                                            dataset=dataset)
        # elif algorithm == 'k-means':
        #     pass
        # elif algorithm == 'dbscan':
        #     pass
        # elif algorithm == 'w2vsim':
        #     w2vsim = W2VSim(file_path_cluwords=path_to_save_cluwords,
        #                     save=False)
        #     self.df_cluwords = w2vsim._create_cluwords(input_vector_file=embedding_file_path,
        #                                                n_words=n_words,
        #                                                n_words_sim=k_neighbors)
        else:
            print('Invalid method')
            exit(0)


class CluwordsTFIDF:
    """
    Description
    -----------
    Calculates Terme Frequency-Inverse Document Frequency (TFIDF) for cluwords.

    Parameters
    ----------
    dataset_file_path : str
        The complete dataset file path.
    n_words : int
        Number of words in the dataset.
    path_to_save_cluwords : list, default None
        Path to save the cluwords file.
    class_file_path: str, (default = None)
        The path to the file with the class of the dataset.

    Attributes
    ----------
    dataset_file_path: str
        The dataset file path passed as parameter.
    path_to_save_cluwords_tfidf: str
        The path to save cluwords passed as parameter, with the addition of the file name.
    n_words: int
        Number of words passed as paramter.
    cluwords_tf_idf: ndarray
        Product between term frequency and inverse term frequency.
    cluwords_idf:

    """

    def __init__(self, dataset, dataset_file_path, n_words, path_to_save_cluwords, class_file_path=None):
        self.dataset_file_path = dataset_file_path
        self.path_to_save_cluwords_tfidf = path_to_save_cluwords + '/cluwords_features.libsvm'
        self.n_words = n_words
        self.cluwords_tf_idf = None
        self.cluwords_idf = None
        loaded = np.load('cluwords_{}.npz'.format(dataset))
        self.vocab = loaded['index']
        self.vocab_cluwords = loaded['cluwords']
        self.cluwords_data = loaded['data']

        self.Y = []
        with open(class_file_path, 'r', encoding="utf-8") as input_file:
            for _class in input_file:
                self.Y.append(np.int(_class))
            input_file.close()
            self.Y = np.asarray(self.Y)

        print('Matrix{}'.format(self.cluwords_data.shape))
        del loaded

        self._read_input()

    def _read_input(self):
        arq = open(self.dataset_file_path, 'r', encoding="utf-8")
        doc = arq.readlines()
        arq.close()

        self.documents = list(map(str.rstrip, doc))
        self.n_documents = len(self.documents)

    def fit_transform(self):
        """Compute cluwords tfidf."""

        # Set number of cluwords
        self.n_cluwords = self.n_words

        # Set vocabulary of cluwords
        self.n_cluwords = len(self.vocab_cluwords)
        print('Number of cluwords {}'.format(len(self.vocab_cluwords)))
        print('Matrix{}'.format(self.cluwords_data.shape))

        print('\nComputing TF...')
        self._cluwords_tf()
        print('\nComputing IDF...')
        self._cluwords_idf()

        self.cluwords_tf_idf = np.multiply(self.cluwords_tf_idf, np.transpose(self.cluwords_idf))

        self._save_tf_idf_features_libsvm()

        return self.cluwords_tf_idf

    def _raw_tf(self, binary=False, dtype=np.float32):
        tf_vectorizer = CountVectorizer(max_features=self.n_words, binary=binary, vocabulary=self.vocab)
        tf = tf_vectorizer.fit_transform(self.documents)

        return np.asarray(tf.toarray(), dtype=dtype)

    def _cluwords_tf(self):
        start = timeit.default_timer()
        tf = self._raw_tf()

        print('tf shape {}'.format(tf.shape))

        self.cluwords_tf_idf = np.zeros((self.n_documents, self.n_cluwords), dtype=np.float16)

        # print('{}'.format())

        self.hyp_aux = []
        for w in range(0, len(self.vocab_cluwords)):
            self.hyp_aux.append(np.asarray(self.cluwords_data[w], dtype=np.float16))
        self.hyp_aux = np.asarray(self.hyp_aux, dtype=np.float32)

        self.cluwords_tf_idf = np.dot(tf, np.transpose(self.hyp_aux))

        # print('Doc')
        # print(tf)

        # print('Cluwords')
        # for w in range(len(hyp_aux)):
        #     print(hyp_aux[w])

        # print('TF')
        # print(self.cluwords_tf)

        end = timeit.default_timer()
        print("Cluwords TF done in %0.3fs." % (end - start))

    def _cluwords_idf(self):
        start = timeit.default_timer()
        print('Read data')
        tf = self._raw_tf(binary=True, dtype=np.float32)
        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        # print('Bin Doc')
        # print(tf)

        start = timeit.default_timer()
        print('Dot tf and hyp_aux')
        ### WITH ERROR ####
        out = np.empty((tf.shape[0], self.hyp_aux.shape[1]), dtype=np.float32)
        ######## CORRECTION #######
        # out = np.empty((tf.shape[0], np.transpose(self.hyp_aux).shape[1]), dtype=np.float32)
        _dot = np.dot(tf, np.transpose(self.hyp_aux), out=out)  # np.array n_documents x n_cluwords
        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        # print('Dot matrix:')
        # print(_dot)

        start = timeit.default_timer()
        print('Divide hyp_aux by itself')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bin_hyp_aux = np.nan_to_num(np.divide(self.hyp_aux, self.hyp_aux))
        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        # print('Bin cluwords')
        # print(bin_hyp_aux)

        start = timeit.default_timer()
        print('Dot tf and bin hyp_aux')
        out = np.empty((tf.shape[0], np.transpose(bin_hyp_aux).shape[1]), dtype=np.float32)
        _dot_bin = np.dot(tf, np.transpose(bin_hyp_aux), out=out)
        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        # print('Count Dot')
        # print(_dot_bin)

        start = timeit.default_timer()
        print('Divide _dot and _dot_bin')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu_hyp = np.nan_to_num(np.divide(_dot, _dot_bin))
        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        # print('Div dot by bin cluwords')
        # print(mu_hyp)

        ##TODO
        # \mu _{c,d} = \frac{1}{\left | \mathcal{V}_{d,c} \right |} \cdot  \sum_{t \in \mathcal{V}_{d,c}} w_t
        #
        ##

        start = timeit.default_timer()
        print('Sum')
        self.cluwords_idf = np.sum(mu_hyp, axis=0)
        end = timeit.default_timer()
        print('Time {}'.format(end - start))

        # print('Mu')
        # print(self.cluwords_idf)

        start = timeit.default_timer()
        print('log')
        self.cluwords_idf = np.log10(np.divide(self.n_documents, self.cluwords_idf))
        end = timeit.default_timer()
        print('Time {}'.format(end - start))
        # print('IDF:')
        # print(self.cluwords_idf)

    def _save_tf_idf_features_libsvm(self):
        tf = self._raw_tf(binary=True, dtype=np.float32)
        with open('{}'.format(self.path_to_save_cluwords_tfidf), 'w', encoding="utf-8") as file:
            for x in range(self.cluwords_tf_idf.shape[0]):
                file.write('{} '.format(self.Y[x]))
                for y in range(1, self.cluwords_tf_idf.shape[1]):
                    if tf[x][y]:
                        file.write('{}:{} '.format(y + 1, self.cluwords_tf_idf[x][y]))
                file.write('\n')
            file.close()
