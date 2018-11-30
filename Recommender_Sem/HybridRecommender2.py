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

        self.URM_train=URM_train
        self.URM_train = check_matrix(URM_train, 'csr')



    def fit(self, ICM_Art=None, ICM_Alb=None, item=None, user=None, SLIM=None, w_itemcf=1.1, w_usercf=0.6, w_cbart=0.3, w_cbalb=0.6, w_slim=0.4, w_svd=0.6, w_rp3=1.1):

        self.item = item

        self.user = user

        self.SLIM = SLIM


        self.CBArt = ItemKNNCBFRecommender(ICM=ICM_Art, URM_train=self.URM_train)

        self.CBAlb = ItemKNNCBFRecommender(ICM=ICM_Alb, URM_train=self.URM_train)

        self.SVD = PureSVDRecommender(URM_train=self.URM_train)

        self.p3 = RP3betaRecommender(URM_train=self.URM_train)

        self.p3.fit(alpha=0.7091718304597212, beta=0.264005617987932, min_rating=0, topK=150, implicit=False, normalize_similarity=True)

        self.SVD.fit(num_factors=490)

        self.simAlb = self.CBAlb.fit(topK=500, shrink=0, similarity='cosine', normalize=True, feature_weighting='none')

        self.simArt = self.CBArt.fit(topK=800, shrink=1000, similarity='cosine', normalize=True, feature_weighting='none')

        self.w_itemcf = w_itemcf

        self.w_usercf = w_usercf

        self.w_cbart = w_cbart

        self.w_cbalb = w_cbalb

        self.w_slim = w_slim

        self.w_svd = w_svd

        self.w_p3 = w_rp3

        # nItems = self.URM_train.shape[1]
        # URMidf = sps.lil_matrix((self.URM_train.shape[0], self.URM_train.shape[1]))
        #
        # for i in range(0, self.URM_train.shape[0]):
        #     IDF_i = log(nItems / np.sum(self.URM_train[i]))
        #     URMidf[i] = np.multiply(self.URM_train[i], IDF_i)
        #
        # self.URM_train = URMidf.tocsr()






    def compute_item_score(self, user_id):

        return normalize(self.item.compute_item_score(user_id)) * self.w_itemcf + normalize(self.user.compute_item_score(user_id))*self.w_usercf + normalize(self.CBAlb.compute_item_score(user_id)) * self.w_cbalb + normalize(self.CBArt.compute_item_score(user_id)) * self.w_cbart + normalize(self.SVD.compute_item_score(user_id))*self.w_svd + normalize(self.p3.compute_item_score(user_id))*self.w_p3 + normalize(self.SLIM.compute_item_score(user_id)) * self.w_slim


    def saveModel(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        dictionary_to_save = {"sparse_weights": self.sparse_weights}





        print("{}: Saving complete".format(self.RECOMMENDER_NAME))