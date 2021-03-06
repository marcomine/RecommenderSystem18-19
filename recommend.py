# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 13:54:54 2018

@author: marco
"""
import numpy as np
import scipy.sparse as sps
from math import log
from cosineSimilarity import Compute_Similarity_Python


class ItemCBFKNNRecommender(object):

    
    def __init__(self, URM, ICM_art, ICM_Alb, ICMU_Alb):     
        self.URM = URM
        self.ICM_art = ICM_art
        self.ICM_Alb = ICM_Alb
        #self.ICMU_art = ICMU_art
        self.ICMU_Alb = ICMU_Alb
        #self.ICM_Dur = ICM_Dur
        
        
        
            
    def fit(self, topK=160, shrink=22, normalize = True):
        
        
        
        
        similarity_object_CF = Compute_Similarity_Python(self.URM, shrink=shrink, 
                                                  topK=200, normalize=normalize, 
                                                  similarity = "cosine")
                                                  
        
        
        self.W_sparse_CF = similarity_object_CF.compute_similarity()
        
        similarity_object_CF_user = Compute_Similarity_Python(self.URM.T, shrink=shrink, 
                                                  topK=70, normalize=normalize, 
                                                  similarity = "cosine")
                                                  
        
        
        self.W_sparse_CF_user = similarity_object_CF_user.compute_similarity()
        
        
        similarity_object_artist = Compute_Similarity_Python(self.ICM_art.T, shrink=5, 
                                                  topK=topK, normalize=normalize, 
                                                  similarity = "cosine")
                                                  
        
        
        self.W_sparse_art = similarity_object_artist.compute_similarity()
        
        similarity_object_album = Compute_Similarity_Python(self.ICM_Alb.T, shrink=5, 
                                                  topK=topK, normalize=normalize, 
                                                  similarity = "cosine")
                                                  
        
        
        self.W_sparse_alb = similarity_object_album.compute_similarity()
        
        #similarity_object_userartist = Compute_Similarity_Python(self.ICMU_art.T, shrink=5, 
        #                                          topK=topK, normalize=normalize, 
        #                                          similarity = "cosine")
                                                  
        
        
        #self.W_sparse_userart = similarity_object_userartist.compute_similarity()
        
        similarity_object_useralbum = Compute_Similarity_Python(self.ICMU_Alb.T, shrink=5, 
                                                  topK=topK, normalize=normalize, 
                                                  similarity = "cosine")
                                                  
        
        
        self.W_sparse_useralb = similarity_object_useralbum.compute_similarity()
        
        nItems = self.URM.shape[1]
        URMidf = sps.lil_matrix((self.URM.shape[0], self.URM.shape[1]))

        for i in range(0, self.URM.shape[0]):
            IDF_i = log(nItems/np.sum(self.URM[i]))
            URMidf[i] = np.multiply(self.URM[i], IDF_i)
           
        self.URM = URMidf.tocsr()
        
       # similarity_object_dur = Compute_Similarity_Python(self.ICM_Dur.T, shrink=shrink, 
      #                                            topK=topK, normalize=normalize, 
       #                                           similarity = similarity)
                                                  
        
        
      #  self.W_sparse_dur = similarity_object_dur.compute_similarity()
      
      
        self.URM_CF = self.URM.dot(self.W_sparse_CF)
        self.URM_art = self.URM.dot(self.W_sparse_art)
        self.URM_alb = self.URM.dot(self.W_sparse_alb)
        
        self.URM_CF_user = self.W_sparse_CF_user.dot(self.URM)
        #self.URM_artuser = self.W_sparse_userart.dot(self.URM)
        self.URM_albuser = self.W_sparse_useralb.dot(self.URM)
        
        
        self.URM_final_hybrid = self.URM_CF*1.1 + (self.URM_art*0.5 + self.URM_alb*1 )*0.6  + self.URM_CF_user*0.6 + self.URM_albuser*0.4       
        

        
    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        
        scores = self.URM_final_hybrid[user_id].toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]
            
        return ranking[:at]
    
    
    def filter_seen(self, user_id, scores):

        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id+1]

        user_profile = self.URM.indices[start_pos:end_pos]
        
        scores[user_profile] = -np.inf

        return scores