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


class HybridRecommender(Recommender):


    def __init__(self, URM_train):

        super(HybridRecommender, self).__init__()

        self.URM_train=URM_train
        self.URM_train = check_matrix(URM_train, 'csr')

    def fit(self, ICM_Art, ICM_Alb, w_itemcf=1.1, w_usercf=0.6, w_cbart=0.3, w_cbalb=0.6, w_slim=0.4, w_svd=0.6):

        self.w_itemcf = w_itemcf

        self.w_usercf = w_usercf

        self.w_cbart = w_cbart

        self.w_cbalb = w_cbalb

        self.w_slim = w_slim

        self.w_svd = w_svd

        self.item = ItemKNNCFRecommender(self.URM_train)

        self.user = UserKNNCFRecommender(self.URM_train)

        self.CBArt = ItemKNNCBFRecommender(ICM=ICM_Art, URM_train=self.URM_train)

        self.CBAlb = ItemKNNCBFRecommender(ICM=ICM_Alb, URM_train=self.URM_train)

        self.SLIM = SLIMElasticNetRecommender(URM_train=self.URM_train)

        self.SVD = PureSVDRecommender(URM_train=self.URM_train)

        self.item.fit(topK=800, shrink=10, similarity='cosine', normalize=True)

        self.user.fit(topK=400, shrink=0, similarity='cosine', normalize=True)

        self.CBAlb.fit(topK=160, shrink=5, similarity='cosine', normalize=True, feature_weighting='none')

        self.CBArt.fit(topK=160, shrink=5, similarity='cosine', normalize=True, feature_weighting='none')

        self.SLIM.fit(l1_penalty=1e-05, l2_penalty=0, positive_only=True, topK=150, alpha = 0.00415637376180466)




    def compute_item_score(self, user_id):

        return self.item.compute_item_score(user_id) * self.w_itemcf + self.user.compute_item_score(user_id)*self.w_usercf + self.CBAlb.compute_item_score(user_id)*self.w_cbalb + self.CBArt.compute_item_score(user_id)*self.w_cbart +self.SLIM.compute_item_score(user_id)*self.w_slim #+ self.SVD.compute_item_score(user_id)*self.w_svd
