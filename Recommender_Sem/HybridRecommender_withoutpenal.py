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



    def fit(self, ICM_Art=None, ICM_Alb=None, item=None, user=None, SLIM=None, p3_alpha=None, users=0, target=[], topPop=0, w_itemcf=1.1, w_usercf=0.6, w_cbart=0.3, w_cbalb=0.6, w_slim=1.0, w_svd=0.6, w_rp3=1.1, w_p3_alpha=1.1, cb_weight=1.0, cf_weight=1.0, graph_weight=1.0, hybrid_weight=1.0, final_weight=1.0, topKItem = 800, topKUser=400, shrinkItem=22, shrinkUser=0, topKP3=200, alphaP3= 0.7312418567825512, topKRP3 = 150, alphaRP3 = 0.8729488414975284, betaRP3= 0.2541372492523202, factorsSVD= 490, topKCBAlb= 500, topKCBArt=800, shrinkCBAlb = 0, shrinkCBArt=1000, pen_offset=None):

        self.item = item

        self.user = user

        self.SLIM = SLIM

        self.p3_alpha = p3_alpha

        self.final_weight = final_weight

        self.target = target

        self.p3_alpha.fit(topK=topKP3, alpha=alphaP3, normalize_similarity=True)



        # topPopRec = TopPop(self.URM_train)
        # topPopRec.fit()
        #
        # self.filterTopPop_ItemsID = topPopRec.recommend(0, cutoff=topPop)


        self.item.fit(topK=topKItem, shrink=shrinkItem, similarity='cosine', normalize=True)



        self.user.fit(topK=topKUser, shrink=shrinkUser, similarity='cosine', normalize=True)




        self.CBArt = ItemKNNCBFRecommender(ICM=ICM_Art, URM_train=self.URM_train)

        self.CBAlb = ItemKNNCBFRecommender(ICM=ICM_Alb, URM_train=self.URM_train)

        self.SVD = PureSVDRecommender(URM_train=self.URM_train)

        self.p3 = RP3betaRecommender(URM_train=self.URM_train)

        self.p3.fit(alpha=alphaRP3, beta= betaRP3, min_rating=0, topK=topKRP3, implicit=True, normalize_similarity=True)

        self.SVD.fit(num_factors=factorsSVD)

        self.CBAlb.fit(topK=topKCBAlb, shrink=shrinkCBAlb, similarity='cosine', normalize=True, feature_weighting='none')

        self.CBArt.fit(topK=topKCBArt, shrink=shrinkCBArt, similarity='cosine', normalize=True, feature_weighting='none')

        self.w_itemcf = w_itemcf

        self.w_usercf = w_usercf

        self.w_cbart = w_cbart

        self.w_cbalb = w_cbalb

        self.w_slim = w_slim

        self.w_svd = w_svd

        self.w_p3 = w_rp3

        self.w_p3_alpha = w_p3_alpha

        # if(idf):
        #
        #     nItems = self.URM_train.shape[1]
        #     URMidf = sps.lil_matrix((self.URM_train.shape[0], self.URM_train.shape[1]))
        #
        #     for i in range(0, self.URM_train.shape[0]):
        #         IDF_i = log(nItems / np.sum(self.URM_train[i]))
        #         URMidf[i] = np.multiply(self.URM_train[i], IDF_i)
        #
        #     self.URM_train = URMidf.tocsr()

        # simCB = simAlb * self.w_cbalb + simArt * self.w_cbart + self.item * self.w_itemcf + self.SLIM * self.w_slim
        # simCB = self.URM_train.dot(simCB)
        #
        # simGraph = simRP3*self.w_p3 + p3_alpha*self.p3_alpha
        # simGraph = self.URM_train.dot(simGraph)
        #
        # #simCF =
        # #simCF = self.URM_train.dot(simCF)
        #
        # self.final_hybrid = simCB * cb_weight + simGraph*graph_weight #+ simCF * cf_weight
        # print(type(self.final_hybrid))
        #
        # #simUserCF = user.dot(self.URM_train)
        #
        #
        # print(type(self.final_hybrid))

    #+ sps.csr_matrix(self.p3_alpha.compute_item_score(user_id)) * self.w_p3_alpha

        # self.pen_array = []
        # for i in range(10):
        #     self.pen_array.append(1-(pen_offset * i))
        #
        #
        #self.final_scores_matrix = self.compute_score_matrix(range(users))



        # if (idf):
        #
        #     nItems = self.final_scores_matrix.shape[1]
        #     URMidf = sps.lil_matrix((self.final_scores_matrix.shape[0], self.final_scores_matrix.shape[1]))
        #
        #     for i in range(0, self.final_scores_matrix.shape[0]):
        #         IDF_i = np.log(nItems / np.sum(self.final_scores_matrix[i]))
        #         URMidf[i] = np.multiply(self.final_scores_matrix[i], IDF_i)
        #
        #     self.final_scores_matrix = URMidf.tocsr()



    def compute_item_score(self, user_id):

        #recs = self.final_hybrid[user_id] * self.final_weight + sps.csr_matrix(self.SVD.compute_item_score(user_id)) * self.w_svd + sps.csr_matrix(self.user.compute_item_score(user_id)) * self.w_usercf

        #recs = sps.csr_matrix(self.item.compute_item_score(user_id)) * self.w_itemcf + sps.csr_matrix(self.CBArt.compute_item_score(user_id)) * self.w_cbart + sps.csr_matrix(self.CBAlb.compute_item_score(user_id)) * self.w_cbalb + sps.csr_matrix(self.p3.compute_item_score(user_id)) * self.w_p3  + sps.csr_matrix(self.SVD.compute_item_score(user_id)) * self.w_svd + sps.csr_matrix(self.user.compute_item_score(user_id)) * self.w_usercf  + sps.csr_matrix(self.p3_alpha.compute_item_score(user_id)) * self.w_p3_alpha+ sps.csr_matrix(self.SLIM.compute_item_score(user_id)) * self.w_slim

        #recs = self.final_scores_matrix[user_id]

        #only slim and ucf and cb

        recs =  sps.csr_matrix(
            self.CBArt.compute_item_score(user_id)) * self.w_cbart + sps.csr_matrix(
            self.CBAlb.compute_item_score(user_id)) * self.w_cbalb + sps.csr_matrix(
            self.user.compute_item_score(user_id)) * self.w_usercf  + sps.csr_matrix(
            self.SLIM.compute_item_score(user_id)) * self.w_slim

        print(type(recs))

        return recs.toarray()



    # def compute_score_matrix(self, user_id):
    #
    #     #recs = self.final_hybrid[user_id] * self.final_weight + sps.csr_matrix(self.SVD.compute_item_score(user_id)) * self.w_svd + sps.csr_matrix(self.user.compute_item_score(user_id)) * self.w_usercf
    #
    #     #recs = self.item.compute_item_score(user_id) * self.w_itemcf #+ self.CBArt.compute_item_score(user_id) * self.w_cbart + self.CBAlb.compute_item_score(user_id) * self.w_cbalb + self.p3.compute_item_score(user_id) * self.w_p3  + self.SVD.compute_item_score(user_id) * self.w_svd + self.user.compute_item_score(user_id) * self.w_usercf  #+ sps.csr_matrix(self.SLIM.compute_item_score(user_id)) * self.w_slim
    #
    #
    #
    #     recs = self.penalize(recs, self.pen_array)
    #
    #
    #     return recs

    # def recommend(self, user_id_array, cutoff = None, remove_seen_flag=True, remove_top_pop_flag = False, remove_CustomItems_flag = False):
    #
    #     # If is a scalar transform it in a 1-cell array
    #     if np.isscalar(user_id_array):
    #         user_id_array = np.atleast_1d(user_id_array)
    #         single_user = True
    #     else:
    #         single_user = False
    #
    #
    #     if cutoff is None:
    #         cutoff = self.URM_train.shape[1] - 1
    #
    #     # Compute the scores using the model-specific function
    #     # Vectorize over all users in user_id_array
    #     scores_batch = self.compute_item_score(user_id_array)
    #
    #
    #     # if self.normalize:
    #     #     # normalization will keep the scores in the same range
    #     #     # of value of the ratings in dataset
    #     #     user_profile = self.URM_train[user_id]
    #     #
    #     #     rated = user_profile.copy()
    #     #     rated.data = np.ones_like(rated.data)
    #     #     if self.sparse_weights:
    #     #         den = rated.dot(self.W_sparse).toarray().ravel()
    #     #     else:
    #     #         den = rated.dot(self.W).ravel()
    #     #     den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
    #     #     scores /= den
    #
    #
    #     for user_index in range(len(user_id_array)):
    #
    #         user_id = user_id_array[user_index]
    #
    #         if remove_seen_flag:
    #             scores_batch[user_index,:] = self._remove_seen_on_scores(user_id, scores_batch[user_index, :])
    #
    #         # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
    #         # - Partition the data to extract the set of relevant items
    #         # - Sort only the relevant items
    #         # - Get the original item index
    #         # relevant_items_partition = (-scores_user).argpartition(cutoff)[0:cutoff]
    #         # relevant_items_partition_sorting = np.argsort(-scores_user[relevant_items_partition])
    #         # ranking = relevant_items_partition[relevant_items_partition_sorting]
    #         #
    #         # ranking_list.append(ranking)
    #
    #
    #     if remove_top_pop_flag:
    #         scores_batch = self._remove_TopPop_on_scores(scores_batch)
    #
    #     if remove_CustomItems_flag:
    #         scores_batch = self._remove_CustomItems_on_scores(scores_batch)
    #
    #     # scores_batch = np.arange(0,3260).reshape((1, -1))
    #     # scores_batch = np.repeat(scores_batch, 1000, axis = 0)
    #
    #     # relevant_items_partition is block_size x cutoff
    #     relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:,0:cutoff]
    #
    #     # Get original value and sort it
    #     # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
    #     # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
    #     relevant_items_partition_original_value = scores_batch[np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
    #     relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
    #     ranking = relevant_items_partition[np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]
    #
    #     ranking_list = ranking.tolist()
    #
    #
    #     # Return single list for one user, instead of list of lists
    #     if single_user:
    #         ranking_list = ranking_list[0]
    #
    #     return ranking_list




    def saveModel(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        dictionary_to_save = {"sparse_weights": self.sparse_weights}





        print("{}: Saving complete".format(self.RECOMMENDER_NAME))