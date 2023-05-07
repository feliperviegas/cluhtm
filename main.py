import os
from optparse import OptionParser
from script_functions import create_embedding_models
from script_functions import generate_topics
from script_functions import save_cluword_representation


def main():
    parser = OptionParser(usage="usage: %prog [options] corpus_file")
    parser.add_option("-d", "--dataset", action="store", type="string", dest="dataset",
                      help="base output directory (default is current directory)", default=None)
    options, args = parser.parse_args()
    # Paths and files paths
    MAIN_PATH = "/home/local/FARFETCH/felipe.viegas/repo/cluhtm"
    EMBEDDING_RESULTS = "fasttext_wiki"
    PATH_TO_SAVE_RESULTS = f"{MAIN_PATH}/{EMBEDDING_RESULTS}/results"
    PATH_TO_SAVE_MODEL = f"{MAIN_PATH}/{EMBEDDING_RESULTS}/datasets/gn_w2v_models"
    EMBEDDINGS_BIN_TYPE = False
    DATASET = options.dataset
    N_THREADS = 6
    ALGORITHM_TYPE = "knn_cosine"

    DATASETS_PATH = f"/{MAIN_PATH}/textual_folds/{DATASET}Pre.txt"
    CLASS_PATH = f"/{MAIN_PATH}/textual_folds/{DATASET}Class.txt"
    # EMBEDDINGS_FILE_PATH = '/{}/wiki-news-300d-1M.vec'.format(MAIN_PATH)
    EMBEDDINGS_FILE_PATH = f"/{MAIN_PATH}/embedding_viegas_concat/embedding_{DATASET}Pre_bert_concat_avg"
    # EMBEDDINGS_DIMENSION = 300
    EMBEDDINGS_DIMENSION = 3072
    SEED=45
    # EMBEDDINGS_FILE_PATH = '{}/viegas_dataset_avg_max/embedding_{}Pre_avg'.format(MAIN_PATH,
    #                                                                               DATASET)

    try:
        os.mkdir('{}/{}'.format(MAIN_PATH, EMBEDDING_RESULTS))
        os.mkdir('{}/{}/results'.format(MAIN_PATH, EMBEDDING_RESULTS))
        os.mkdir('{}/{}/datasets'.format(MAIN_PATH, EMBEDDING_RESULTS))
        os.mkdir('{}/{}/datasets/gn_w2v_models'.format(MAIN_PATH, EMBEDDING_RESULTS))
    except FileExistsError:
        pass

    # Create the word2vec models for each dataset
    print('Filter embedding space to {} dataset...'.format(DATASET))
    n_words = create_embedding_models(dataset=DATASET,
                                      embedding_file_path=EMBEDDINGS_FILE_PATH,
                                      embedding_dimension=EMBEDDINGS_DIMENSION,
                                      embedding_type=EMBEDDINGS_BIN_TYPE,
                                      datasets_path=DATASETS_PATH,
                                      path_to_save_model=PATH_TO_SAVE_MODEL)

    print('Build topics...')
    generate_topics(dataset=DATASET,
                    word_count=n_words,
                    path_to_save_model=PATH_TO_SAVE_MODEL,
                    datasets_path=DATASETS_PATH,
                    path_to_save_results=PATH_TO_SAVE_RESULTS,
                    n_threads=N_THREADS,
                    algorithm_type=ALGORITHM_TYPE,
                    k=500,
                    threshold=0.4,
                    class_path=CLASS_PATH,
                    seed=SEED)

    save_cluword_representation(dataset=DATASET,
                                word_count=n_words,
                                path_to_save_model=PATH_TO_SAVE_MODEL,
                                datasets_path=DATASETS_PATH,
                                path_to_save_results=PATH_TO_SAVE_RESULTS,
                                n_threads=N_THREADS,
                                algorithm_type=ALGORITHM_TYPE,
                                k=500,
                                threshold=0.4,
                                class_path=CLASS_PATH)


if __name__ == '__main__':
    main()
