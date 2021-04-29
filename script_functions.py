# Packages
import os
import shutil
import pandas as pd
import numpy as np
import logging as log
import glob
import operator
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from cluwords import Cluwords, CluwordsTFIDF
from metrics import Evaluation
from embedding import CreateEmbeddingModels
from generate_nmf import GenerateNFM
from reference_nmf import ReferenceNFM
from topic_stability import TopicStability
from text.util import save_corpus, load_corpus
from collections import deque


def top_words(model, feature_names, n_top_words):
    topico = []
    for topic_idx, topic in enumerate(model.components_):
        top = ''
        top2 = ''
        top += ' '.join([feature_names[i]
                         for i in topic.argsort()[:-n_top_words - 1:-1]])
        top2 += ''.join(str(sorted(topic)[:-n_top_words - 1:-1]))

        topico.append(str(top))

    return topico


def print_results(model, tfidf_feature_names, cluwords_freq, cluwords_docs,
                  dataset, path_to_save_results, path_to_save_model, sufix):
    print(path_to_save_results)
    for t in [5, 10, 20]:
        with open('{}/{}_result_topic_{}.txt'.format(path_to_save_results, sufix, t), 'w', encoding="utf-8") as f_res:
            f_res.write('Topics {}\n'.format(t))
            topics = top_words(model, tfidf_feature_names, t)
            f_res.write('{}\n'.format(topics))
            log.info('Coherence Metric...')
            coherence = Evaluation.coherence(topics, cluwords_freq, cluwords_docs)
            f_res.write('Coherence: {} ({})\n'.format(np.round(np.mean(coherence), 4), np.round(np.std(coherence), 4)))
            f_res.write('{}\n'.format(coherence))
            log.info('PMI Metric...')
            for word, freq in cluwords_freq.items():
                log.info('w {} f {}'.format(word, freq))

            pmi, npmi = Evaluation.pmi(topics, cluwords_freq, cluwords_docs,
                                       sum([freq for word, freq in cluwords_freq.items()]), t)
            f_res.write('PMI: {} ({})\n'.format(np.round(np.mean(pmi), 4), np.round(np.std(pmi), 4)))
            f_res.write('{}\n'.format(pmi))
            f_res.write('NPMI: {} ({})\n'.format(np.round(np.mean(npmi), 4), np.round(np.std(npmi), 4)))
            f_res.write('{}\n'.format(npmi))
            log.info('W2V-L1 Metric...')
            w2v_l1 = Evaluation.w2v_metric(topics, t, path_to_save_model, 'l1_dist', dataset)
            f_res.write('W2V-L1: {} ({})\n'.format(np.round(np.mean(w2v_l1), 4), np.round(np.std(w2v_l1), 4)))
            f_res.write('{}\n'.format(w2v_l1))

            f_res.close()


def save_results(model, tfidf_feature_names, path_to_save_model, dataset, cluwords_freq,
                 cluwords_docs, path_to_save_results):
    res_mean = []
    coherence_mean = ['coherence']
    lcp_mean = ['lcp']
    npmi_mean = ['npmi']
    w2v_l1_mean = ['w2v-l1']

    for t in [5, 10, 20]:
        topics = top_words(model, tfidf_feature_names, t)

        # Write topics in a file
        file = open('{}/topics_{}.txt'.format(path_to_save_results, t), 'w+', encoding="utf-8")
        file.write('TOPICS WITH {} WORDS\n\n'.format(t))
        for i, topic in enumerate(topics):
            file.write('Topic %d\n' % i)
            file.write('%s\n' % topic)
        file.close()

        coherence = Evaluation.coherence(topics, cluwords_freq, cluwords_docs)
        coherence_mean.extend(['{0:.3f} +- {1:.3f}'.format(np.mean(coherence),
                                                           np.std(coherence))])

        lcp = Evaluation.lcp(topics, cluwords_freq, cluwords_docs)
        lcp_mean.extend(['{0:.3f} +- {1:.3f}'.format(np.mean(lcp),
                                                     np.std(lcp))])

        pmi, npmi = Evaluation.pmi(topics, cluwords_freq, cluwords_docs,
                                 sum([freq for word, freq in cluwords_freq.items()]), t)
        npmi_mean.extend(['{0:.3f} +- {1:.3f}'.format(np.mean(npmi),
                                                      np.std(npmi))])

        w2v_l1 = Evaluation.w2v_metric(topics, t, path_to_save_model, 'l1_dist', dataset)
        w2v_l1_mean.extend(['{0:.3f} +- {1:.3f}'.format(np.mean(w2v_l1),
                                                        np.std(w2v_l1))])
    res_mean.extend([coherence_mean, lcp_mean, npmi_mean, w2v_l1_mean])

    df_mean = pd.DataFrame(res_mean, columns=['metric', '5 words', '10 words', '20 words'])

    df_mean.to_csv(path_or_buf='{}/results.csv'.format(path_to_save_results))


def create_embedding_models(dataset, embedding_file_path, embedding_type, datasets_path, path_to_save_model):
    # Create the word2vec models for each dataset
    word2vec_models = CreateEmbeddingModels(embedding_file_path=embedding_file_path,
                                            embedding_type=embedding_type,
                                            document_path=datasets_path,
                                            path_to_save_model=path_to_save_model)
    n_words = word2vec_models.create_embedding_models(dataset)

    return n_words


def set_cluwords_representation(dataset, out_prefix, X, class_path):
    loaded = np.load('cluwords_{}.npz'.format(dataset))
    terms = loaded['cluwords']
    del loaded
    y = []
    with open(class_path, 'r', encoding="utf-8") as filename:
        for line in filename:
            y.append(line.strip())

    y = np.array(y, dtype=np.int32)
    doc_ids = []
    classes = {}
    doc_id = 0
    for document_class in range(0, y.shape[0]):
        doc_ids.append(doc_id)
        if y[document_class] not in classes:
            classes[y[document_class]] = []

        classes[y[document_class]].append(doc_id)
        doc_id += 1

    save_corpus(out_prefix, X, terms, doc_ids, classes)
    return y


def save_topics(model, tfidf_feature_names, cluwords_tfidf, best_k, topics_documents, y, doc_ids, terms, out_prefix,
                dq, k_max, depth, parent, hierarchy, max_depth):
    topics = top_words(model, tfidf_feature_names, 20)
    for k in range(0, best_k):
        topic = np.argwhere(topics_documents == k)
        topic = topic.ravel()
        cluwords_tfidf_temp = cluwords_tfidf.copy()
        cluwords_tfidf_temp = cluwords_tfidf_temp[topic, :]

        doc_ids_temp = doc_ids.copy()
        doc_ids_temp = doc_ids_temp[topic]

        if depth not in hierarchy:
            hierarchy[depth] = {}

        if parent not in hierarchy[depth]:
            hierarchy[depth][parent] = {}

        hierarchy[depth][parent][k] = topics[k]
        classes = {}
        if len(doc_ids_temp) > k_max  and depth+1 < max_depth:
        # if depth < max_depth:
            log.info("Add topic: {} Shape Matrix: {}".format(k, cluwords_tfidf_temp.shape))
            log.info("len(doc_ids): {}".format(len(doc_ids_temp)))
            for doc_id in doc_ids:
                if y[doc_id] not in classes:
                    classes[y[doc_id]] = []

                classes[y[doc_id]].append(doc_id)

            prefix = "{prefix} {k}".format(prefix=out_prefix, k=k)
            save_corpus(prefix, csr_matrix(cluwords_tfidf_temp), terms, doc_ids, classes)
            dq.appendleft(prefix)
        # else:
        #     log.info("Exclude topic: {} Shape Matrix: {}".format(k, cluwords_tfidf_temp.shape))
        #     log.info("len(doc_ids): {}".format(len(doc_ids_temp)))

    return dq, hierarchy


def print_herarchical_structure(output, hierarchy, depth=0, parent='-1', son=0):
    # print('{} {} {}'.format(depth, parent, son))
    if depth not in hierarchy:
        return

    if parent not in hierarchy[depth]:
        return

    if son not in hierarchy[depth][parent]:
        return

    tabulation = '\t' * depth
    output.write('{}{}\n'.format(tabulation, hierarchy[depth][parent][son]))
    print_herarchical_structure(output, hierarchy, depth=depth + 1, parent='{} {}'.format(parent, son),
                                son=0)
    print_herarchical_structure(output, hierarchy, depth=depth, parent=parent, son=son + 1)
    return


def generate_topics(dataset, word_count, path_to_save_model, datasets_path,
                    path_to_save_results, n_threads, k, threshold, class_path, algorithm_type, debug=3):
    log.basicConfig(filename="{}.log".format(dataset), filemode="w", level=max(50 - (debug * 10), 10),
                    format='%(asctime)-18s %(levelname)-10s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d/%m/%Y %H:%M', )
    # Path to files and directories
    embedding_file_path = """{}/{}.txt""".format(path_to_save_model, dataset)
    path_to_save_results = '{}/{}'.format(path_to_save_results, dataset)
    # path_to_save_pkl = '{}/pkl'.format(path_to_save_results)

    try:
        os.mkdir('{}'.format(path_to_save_results))
    except FileExistsError:
        pass

    Cluwords(algorithm=algorithm_type,
             embedding_file_path=embedding_file_path,
             n_words=word_count,
             k_neighbors=k,
             threshold=threshold,
             n_jobs=n_threads,
             dataset=dataset
             )

    cluwords = CluwordsTFIDF(dataset=dataset,
                             dataset_file_path=datasets_path,
                             n_words=word_count,
                             path_to_save_cluwords=path_to_save_results,
                             class_file_path=class_path)
    log.info('Computing TFIDF...')
    cluwords_tfidf = cluwords.fit_transform()
    cluwords_tfidf_temp = cluwords_tfidf.copy()
    cluwords_tfidf_temp = csr_matrix(cluwords_tfidf_temp)  # Convert the cluwords_tfidf array matrix to a sparse cluwords
    # RANGE OF TOPICS THAT WILL BE EXPLOIT BY THE STRATEGY
    k_min = 5
    k_max = 20
    n_runs = 3
    max_depth = 3
    sufix = "{dataset}_{depth}_{parent_topic}".format(dataset=dataset, depth=0, parent_topic='-1')
    y = set_cluwords_representation(dataset,
                                    sufix,
                                    cluwords_tfidf_temp,
                                    class_path)
    dq = deque([sufix])
    hierarchy = {}
    while dq:
        log.info("Deque {}".format(dq))
        log.info("Documents {}".format(dataset))
        sufix = dq.pop()
        log.info("Starting iteration {sufix}".format(sufix=sufix))
        parent = sufix.split("_")[-1]
        depth = int(sufix.split("_")[1])
        log.info("Depth {}".format(depth))
        log.info("Parent Topic {}".format(parent))
        log.info("Reference NMF")
        ReferenceNFM().run(dataset=dataset,
                           corpus_path="{}.pkl".format(sufix),
                           dir_out_base="reference-{}".format(sufix),
                           kmin=k_min,
                           maxiter=10,
                           kmax=k_max)
        log.info("Generate NMF")
        GenerateNFM().run(dataset=dataset,
                          corpus_path="{}.pkl".format(sufix),
                          dir_out_base="topic-{}".format(sufix),
                          kmin=k_min,
                          kmax=k_max,
                          maxiter=10,
                          runs=n_runs)
        log.info("Topic Stability")
        dict_stability = {}
        for k in range(k_min, k_max+1):
            stability = TopicStability().run(dataset=dataset,
                                             reference_rank_path="reference-{}/nmf_k{:02}/ranks_reference.pkl"
                                             .format(sufix, k),
                                             rank_paths=glob.glob("topic-{}/nmf_k{:02}/ranks*".format(sufix, k)),
                                             top=10)
            dict_stability[k] = stability

        best_k = max(dict_stability.keys(), key=(lambda key: dict_stability[key]))
        log.info("Selected K {k} => Stability({k}) = {stability} (median)".format(k=best_k,
                                                                                  stability=round(dict_stability[best_k],
                                                                                                  4)))
        X, terms, doc_ids, classes = load_corpus("{}.pkl".format(sufix))
        # Fit the NMF model
        log.info("\nFitting the NMF model (Frobenius norm) with tf-idf features, shape {}...".format(X.shape))
        nmf = NMF(n_components=best_k,
                  init=None,
                  random_state=1,
                  alpha=.1,
                  l1_ratio=.5).fit(X)

        w = nmf.fit_transform(X)  # matrix W = m x k
        tfidf_feature_names = list(cluwords.vocab_cluwords)
        topics_documents = np.argmax(w, axis=1)

        log.info("\n>>X shape = {}".format(X.shape))

        dq, hierarchy = save_topics(model=nmf,
                                    tfidf_feature_names=tfidf_feature_names,
                                    cluwords_tfidf=X.toarray(),
                                    best_k=best_k,
                                    topics_documents=topics_documents,
                                    y=y,
                                    doc_ids=np.array(doc_ids),
                                    terms=terms,
                                    out_prefix="{dataset}_{depth}_{parent_topic}".format(dataset=dataset,
                                                                                         depth=depth+1,
                                                                                         parent_topic=parent),
                                    dq=dq,
                                    k_max=k_max,
                                    depth=depth,
                                    parent=parent,
                                    hierarchy=hierarchy,
                                    max_depth=max_depth)

        shutil.rmtree("reference-{}".format(sufix))
        shutil.rmtree("topic-{}".format(sufix))
        # os.remove("{}.pkl".format(sufix))

        log.info('End Iteration...')

    log.info(hierarchy)

    output = open('{}/hierarchical_struture.txt'.format(path_to_save_results), 'w', encoding="utf-8")
    print_herarchical_structure(output=output, hierarchy=hierarchy)
    output.close()


def save_cluword_representation(dataset, word_count, path_to_save_model, datasets_path,
                    path_to_save_results, n_threads, k, threshold,
                    class_path, algorithm_type):
    # Path to files and directories
    embedding_file_path = """{}/{}.txt""".format(path_to_save_model, dataset)
    path_to_save_results = '{}/{}'.format(path_to_save_results, dataset)
    # path_to_save_pkl = '{}/pkl'.format(path_to_save_results)

    try:
        os.mkdir('{}'.format(path_to_save_results))
    except FileExistsError:
        pass

    Cluwords(algorithm=algorithm_type,
             embedding_file_path=embedding_file_path,
             n_words=word_count,
             k_neighbors=k,
             threshold=threshold,
             n_jobs=n_threads,
             dataset=dataset
             )

    cluwords = CluwordsTFIDF(dataset=dataset,
                             dataset_file_path=datasets_path,
                             n_words=word_count,
                             path_to_save_cluwords=path_to_save_results,
                             class_file_path=class_path)
    log.info('Computing TFIDF...')
    cluwords_tfidf = cluwords.fit_transform()
    np.savez_compressed('{}/cluwords_representation_{}.npz'.format(path_to_save_results, dataset),
                        tfidf=cluwords_tfidf,
                        feature_names=cluwords.vocab_cluwords)
