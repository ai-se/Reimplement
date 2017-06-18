from __future__ import division
import pickle


def merge_pickle(rank_pickle, progressive_pickle, projective_pickle):
    pickle_folder = "PickleLocker/"
    rank = pickle.load(open(pickle_folder + rank_pickle, "r"))
    progressive = pickle.load(open(pickle_folder + progressive_pickle, "r"))
    projective = pickle.load(open(pickle_folder + projective_pickle, "r"))

    final = {}
    for key in rank.keys():
        final[key] = {}
        r1 = rank[key]
        prog1 = progressive[key]
        proj1 = projective[key]

        final[key]['rank_train_size'] = r1[0.4]['rank-based']['train_set_size']
        final[key]['progressive_train_size'] = prog1[0.4]['progressive']['train_set_size']
        final[key]['projective_train_size'] = proj1[0.4]['projective']['train_set_size']

        final[key]['rank_min_rank'] = r1[0.4]['rank-based']['min_rank']
        final[key]['progressive_min_rank'] = prog1[0.4]['progressive']['min_rank']
        final[key]['projective_min_rank'] = proj1[0.4]['projective']['min_rank']

    pickle.dump(final, open("./stats/merged.p", "w"))

if __name__ == "__main__":
    merge_pickle("rank_based.p", "progressive_sampling.p", "projective_sampling.p")