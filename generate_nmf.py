#!/usr/bin/env python
import os
import random
import logging as log
from optparse import OptionParser
import numpy as np
import text.util
import unsupervised.nmf
import unsupervised.rankings
import unsupervised.util


class GenerateNFM(object):
    @staticmethod
    def run(dataset, corpus_path, dir_out_base, maxiter=10, kmin=5, kmax=5, sample_ratio=0.8,
            seed=1000, runs=1, write_factors=False, use_nimfa=False, debug=3):
        log.basicConfig(filename="{}_generate.log".format(dataset), filemode="w",
                        level=max(50 - (debug * 10), 10), format='%(asctime)-18s %(levelname)-10s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%d/%m/%Y %H:%M', )
        # Load the cached corpus
        X, terms, doc_ids, classes = text.util.load_corpus(corpus_path)  # type: (csr_matrix, list, list, dict)

        # Choose implementation
        if use_nimfa:
            impl = unsupervised.nmf.NimfaNMF(max_iters=maxiter, init_strategy="random", update="euclidean")
        else:
            impl = unsupervised.nmf.SklNMF(max_iters=maxiter, init_strategy="random")

        n_documents = X.shape[0]
        n_sample = int(sample_ratio * n_documents)
        indices = np.arange(n_documents)

        # Generate all NMF topic models for the specified numbers of topics
        log.info("Testing models in range k=[{},{}]".format(kmin, kmax))
        log.info("Sampling ratio = {} - {}/{} documents per run".format(round(sample_ratio,2), n_sample, n_documents))
        for k in range(kmin, kmax + 1):
            # Set random state
            np.random.seed(seed)
            random.seed(seed)
            log.info("Applying NMF (k={}, runs={}, seed={} - {}) ...".format(k, runs, seed, impl.__class__.__name__))
            dir_out_k = os.path.join(dir_out_base, "nmf_k{:02}".format(k))
            if not os.path.exists(dir_out_k):
                os.makedirs(dir_out_k)
            log.debug("Results will be written to {}".format(dir_out_k))
            # Run NMF
            for r in range(runs):
                log.info("NMF run {}/{} (k={}, max_iters={})".format(r + 1, runs, k, maxiter))
                file_suffix = "{}_{:03}".format(seed, r + 1)
                # sub-sample data
                np.random.shuffle(indices)
                sample_indices = indices[0:n_sample]
                S = X[sample_indices, :]
                sample_doc_ids = []
                for doc_index in sample_indices:
                    sample_doc_ids.append(doc_ids[doc_index])
                # apply NMF
                impl.apply(S, k)
                # Get term rankings for each topic
                term_rankings = []
                for topic_index in range(k):
                    ranked_term_indices = impl.rank_terms(topic_index)
                    term_ranking = [terms[i] for i in ranked_term_indices]
                    term_rankings.append(term_ranking)
                log.debug("Generated ranking set with {} topics covering up to {} terms".
                          format(len(term_rankings),
                                 unsupervised.rankings.term_rankings_size(term_rankings)))
                # Write term rankings
                ranks_out_path = os.path.join(dir_out_k, "ranks_{}.pkl".format(file_suffix))
                log.debug("Writing term ranking set to %s" % ranks_out_path)
                unsupervised.util.save_term_rankings(ranks_out_path, term_rankings)
                # Write document partition
                partition = impl.generate_partition()
                partition_out_path = os.path.join(dir_out_k, "partition_{}.pkl".format(file_suffix))
                log.debug("Writing document partition to {}".format(partition_out_path))
                unsupervised.util.save_partition(partition_out_path, partition, sample_doc_ids)
                # Write the complete factorization?
                if write_factors:
                    factor_out_path = os.path.join(dir_out_k, "factors_{}.pkl".format(file_suffix))
                    # NB: need to make a copy of the factors
                    log.debug("Writing factorization to {}".format(factor_out_path))
                    unsupervised.util.save_nmf_factors(factor_out_path, np.array(impl.W), np.array(impl.H), sample_doc_ids)

