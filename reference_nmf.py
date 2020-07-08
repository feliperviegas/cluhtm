#!/usr/bin/env python
import os
import sys
import random
import logging as log
from optparse import OptionParser
import numpy as np
import text.util
import unsupervised.nmf
import unsupervised.rankings
import unsupervised.util


class ReferenceNFM(object):
    @staticmethod
    def run(dataset, corpus_path, dir_out_base, maxiter=10, kmin=5, kmax=5,seed=1000,
            top=10, write_factors=False, use_nimfa=False, debug=3):
        log.basicConfig(filename="{}_reference.log".format(dataset), filemode="w",
                        level=max(50 - (debug * 10), 10), format='%(asctime)-18s %(levelname)-10s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%d/%m/%Y %H:%M', )
        log_level = max(50 - (debug * 10), 10)
        # Set random state
        np.random.seed(seed)
        random.seed(seed)

        # Load the cached corpus
        log.info("Loading corpus from {}...".format(corpus_path))
        X, terms, doc_ids, classes = text.util.load_corpus(corpus_path)  # type: (csr_matrix, list, list, dict)
        log.debug("Read {} document-term matrix, dictionary of {} terms, list of {} document IDs"
                  .format(str(X.shape), len(terms), len(doc_ids)))

        # Choose implementation
        if use_nimfa:
            impl = unsupervised.nmf.NimfaNMF(max_iters=maxiter, init_strategy="nndsvd", update="euclidean")
        else:
            impl = unsupervised.nmf.SklNMF(max_iters=maxiter, init_strategy="nndsvd")

        # Generate reference NMF topic models for the specified numbers of topics
        log.info("Running reference experiments in range k=[{},{}] max_iters={}"
                 .format(kmin, kmax, maxiter))
        for k in range(kmin, kmax + 1):
            log.info("Applying NMF k={} ({}) ...".format(k, impl.__class__.__name__))
            dir_out_k = os.path.join(dir_out_base, "nmf_k{:02}".format(k))
            if not os.path.exists(dir_out_k):
                os.makedirs(dir_out_k)

            impl.apply(X, k)
            log.debug("Generated W {} and H {}".format(str(impl.W.shape), str(impl.H.shape)))
            # Get term rankings for each topic
            term_rankings = []
            for topic_index in range(k):
                ranked_term_indices = impl.rank_terms(topic_index)
                term_ranking = [terms[i] for i in ranked_term_indices]
                term_rankings.append(term_ranking)

            log.info("Generated %d rankings covering up to {} terms"
                     .format(len(term_rankings), unsupervised.rankings.term_rankings_size(term_rankings)))
            # Print out the top terms, if we want verbose output
            if log_level <= 10 and top > 0:
                print(unsupervised.rankings.format_term_rankings(term_rankings, top=top))

            log.info("Writing results to {}".format(dir_out_k))
            # Write term rankings
            ranks_out_path = os.path.join(dir_out_k, "ranks_reference.pkl")
            log.debug("Writing term ranking set to %s" % ranks_out_path)
            unsupervised.util.save_term_rankings(ranks_out_path, term_rankings)
            # Write document partition
            partition = impl.generate_partition()
            partition_out_path = os.path.join(dir_out_k, "partition_reference.pkl")
            log.debug("Writing document partition to %s" % partition_out_path)
            unsupervised.util.save_partition(partition_out_path, partition, doc_ids)
            # Write the complete factorization?
            if write_factors:
                factor_out_path = os.path.join(dir_out_k, "factors_reference.pkl")
                # NB: need to make a copy of the factors
                log.debug("Writing complete factorization to %s" % factor_out_path)
                unsupervised.util.save_nmf_factors(factor_out_path, np.array(impl.W), np.array(impl.H), doc_ids)
