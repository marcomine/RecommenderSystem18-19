from sklearn.preprocessing import normalize

from Base.Recommender_utils import check_matrix
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender

from Base.NonPersonalizedRecommender import TopPop, Random

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
from Base.Recommender import Recommender
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender

#from data.Movielens_10M.Movielens10MReader import Movielens10MReader
#from DataReader import dataReader
from DataReaderWithoutValid import dataReader

from math import log
import scipy.sparse as sps
import numpy as np


class HybridRecommender(Recommender):
    RECOMMENDER_NAME = "ItemKNNCFRecommender"

    def __init__(self, URM_train):

        super(HybridRecommender, self).__init__()

        self.URM_train = URM_train
        self.URM_train = check_matrix(URM_train, 'csr')
        self.penalized = False



    def fit(self, ICM_Art=None, ICM_Alb=None, item=None, user=None, SLIM=None , reader=None, w_itemcf=1.1, w_usercf=0.6, w_cbart=0.3, w_cbalb=0.6, w_slim=0.4, w_rp3=1.1):





        #self.SLIM = SLIM

        self.reader = reader

        #self.p3_alpha = p3_alpha


        self.CBArt = ItemKNNCBFRecommender(ICM=ICM_Art, URM_train=self.URM_train)

        self.CBAlb = ItemKNNCBFRecommender(ICM=ICM_Alb, URM_train=self.URM_train)

        #self.SVD = PureSVDRecommender(URM_train=self.URM_train)

        self.p3 = RP3betaRecommender(URM_train=self.URM_train)

        simURM_rp3 = self.p3.fit(alpha=0.8729488414975284, beta= 0.2541372492523202, min_rating=0, topK=150, implicit=True, normalize_similarity=True)
        simURM_rp3 = self.URM_train.dot(simURM_rp3)

        #self.SVD.fit(num_factors=490)

        simAlb = self.CBAlb.fit(topK=500, shrink=0, similarity='cosine', normalize=True, feature_weighting='none')
        simAlb = self.URM_train.dot(simAlb)

        simArt = self.CBArt.fit(topK=800, shrink=1000, similarity='cosine', normalize=True, feature_weighting='none')
        simArt = self.URM_train.dot(simArt)

        self.w_itemcf = w_itemcf

        self.w_usercf = w_usercf

        self.w_cbart = w_cbart

        self.w_cbalb = w_cbalb

        self.w_slim = w_slim

        #self.w_svd = w_svd

        self.w_p3 = w_rp3

        #self.w_p3_alpha = w_p3_alpha

        # nItems = self.URM_train.shape[1]
        # URMidf = sps.lil_matrix((self.URM_train.shape[0], self.URM_train.shape[1]))
        #
        # for i in range(0, self.URM_train.shape[0]):
        #     IDF_i = log(nItems / np.sum(self.URM_train[i]))
        #     URMidf[i] = np.multiply(self.URM_train[i], IDF_i)
        #
        # self.URM_train = URMidf.tocsr()




        self.URM_final = item*self.w_itemcf + user*self.w_usercf  + simURM_rp3*self.w_p3 + simArt*self.w_cbart+simAlb*self.w_cbalb#+ self.SLIM*self.w_slim

        self.URM_final = check_matrix(self.URM_final, 'csr')




    def penalize(self, scores_matrix, pen_array):

        submission = {}



        for i in range(10):

            print("Iteration")
            tracks_to_pen = []

            target = self.reader.get_target_list()

            target = set(target)
            print("starting iterate on users")

            for user_id in range(self.URM_train.shape[0]):

                if user_id in target:

                    if (i==0):

                        submission[user_id] = []

                    print("slicing")

                    scores = scores_matrix[user_id]

                    scores = scores.toarray().ravel()

                    print("filtering")

                    scores = self.filter_seen(user_id, scores)

                    ranking = scores.argsort()[::-1]

                    if(i<4):

                        tracks_to_pen.append(ranking[0])

                    submission[user_id].append(ranking[0])

                    scores_matrix[user_id,ranking[0]] = -np.inf


            print("start penalizing")
            if(i<4):
                tracks_to_pen = np.unique(np.array(tracks_to_pen))

                scores_matrix.data[np.in1d(scores_matrix.indices, tracks_to_pen)] *= pen_array[i]

        return submission


    def filter_seen(self, user_id, scores):

        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

    def compute_item_score(self, user_id):

        if (self.penalized == False):
            self.URM_final = self.penalize(self.URM_final, [0.95, 0.90, 0.85, 0.80])
            self.penalized = True

        return self.URM_final[user_id]


    def saveModel(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        dictionary_to_save = {"sparse_weights": self.sparse_weights}





        print("{}: Saving complete".format(self.RECOMMENDER_NAME))