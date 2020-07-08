#!/usr/bin/env python
import os
import sys
import logging as log
from optparse import OptionParser
import numpy as np
import unsupervised.util
import unsupervised.rankings


class TopicStability(object):
    @staticmethod
    def run(reference_rank_path, dataset, rank_paths, top=20, debug=3):
        if len(rank_paths) < 2:
            log.error("Must specify at least two ranking sets")

        log.basicConfig(filename="{}_stability.log".format(dataset), filemode="w",
                        level=max(50 - (debug * 10), 10), format='%(asctime)-18s %(levelname)-10s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%d/%m/%Y %H:%M', )

        # Load cached ranking sets
        log.info("Reading {} term ranking sets (top={}) ...".format(len(rank_paths), top))
        all_term_rankings = []
        for rank_path in rank_paths:
            # first set is the reference set
            if len(all_term_rankings) == 0:
                log.debug("Loading reference term ranking set from {} ...".format(rank_path))
            else:
                log.debug("Loading test term ranking set from {} ...".format(rank_path))
            term_rankings, labels = unsupervised.util.load_term_rankings(rank_path)
            log.debug("Set has {} rankings covering {} terms"
                      .format(len(term_rankings), unsupervised.rankings.term_rankings_size(term_rankings)))
            # do we need to truncate the number of terms in the ranking?
            if top > 1:
                term_rankings = unsupervised.rankings.truncate_term_rankings(term_rankings, top)
                log.debug("Truncated to {} -> set now has {} rankings covering {} terms"
                          .format(top, len(term_rankings), unsupervised.rankings.term_rankings_size(term_rankings)))
            all_term_rankings.append(term_rankings)

        # First argument was the reference term ranking
        term_rankings, labels = unsupervised.util.load_term_rankings(reference_rank_path)
        reference_term_ranking = unsupervised.rankings.truncate_term_rankings(term_rankings, top)

        r = len(all_term_rankings)
        log.info("Loaded {} non-reference term rankings".format(r))

        # Perform the evaluation
        metric = unsupervised.rankings.AverageJaccard()
        matcher = unsupervised.rankings.RankingSetAgreement(metric)
        log.info("Performing reference comparisons with {} ...".format(str(metric)))
        all_scores = []
        for i in range(r):
            score = matcher.similarity(reference_term_ranking, all_term_rankings[i])
            all_scores.append(score)

        # Get overall score across all candidates
        all_scores = np.array(all_scores)
        log.info("Stability={mean} [{min},{max}] => {median}"
                 .format(mean=round(all_scores.mean(), 4),
                         min=round(all_scores.min(), 4),
                         max=round(all_scores.max(), 4),
                         median=np.median(all_scores)))
        # return all_scores.mean()
        return np.median(all_scores, axis=0)
