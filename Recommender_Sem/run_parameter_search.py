#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""
from DataReader_withUCM import dataReader

#from DataReaderWithoutValid import dataReader
from Base.NonPersonalizedRecommender import TopPop, Random
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_ElasticNet
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, \
    MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender
#from MatrixFactorization.PyTorch.MF_MSE_PyTorch import MF_MSE_PyTorch
import  numpy as np



from ParameterTuning.BayesianSearch import BayesianSearch
from ParameterTuning.GridSearch import GridSearch

import traceback, pickle
from Utils.PoolWithSubprocess import PoolWithSubprocess

from HybridRecommender2 import HybridRecommender

from ParameterTuning.AbstractClassSearch import DictionaryKeys

from MatrixFactorization.MatrixFactorization_RMSE import FunkSVD


def run_KNNCFRecommender_on_similarity_type(similarity_type, parameterSearch, URM_train, n_cases, output_root_path,
                                            metric_to_optimize):
    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = [x for x in range(100, 1500, 100)]
    hyperparamethers_range_dictionary["shrink"] = [x for x in range(0, 200, 2)]
    hyperparamethers_range_dictionary["similarity"] = [similarity_type]
    hyperparamethers_range_dictionary["normalize"] = [True]

    if similarity_type == "asymmetric":
        hyperparamethers_range_dictionary["asymmetric_alpha"] = range(0, 2)
        hyperparamethers_range_dictionary["normalize"] = [True]

    elif similarity_type == "tversky":
        hyperparamethers_range_dictionary["tversky_alpha"] = range(0, 2)
        hyperparamethers_range_dictionary["tversky_beta"] = range(0, 2)
        hyperparamethers_range_dictionary["normalize"] = [True]

    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    output_root_path_similarity = output_root_path + "_" + similarity_type

    best_parameters = parameterSearch.search(recommenderDictionary,
                                             n_cases=n_cases,
                                             output_root_path=output_root_path_similarity,
                                             metric=metric_to_optimize)


def run_KNNCBFRecommender_on_similarity_type(similarity_type, parameterSearch, URM_train, ICM_train, n_cases,
                                             output_root_path, metric_to_optimize):
    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
    hyperparamethers_range_dictionary["shrink"] = [0, 10, 50, 100, 200, 300, 500, 1000]
    hyperparamethers_range_dictionary["similarity"] = [similarity_type]
    hyperparamethers_range_dictionary["normalize"] = [True, False]

    if similarity_type == "asymmetric":
        hyperparamethers_range_dictionary["asymmetric_alpha"] = range(0, 2)
        hyperparamethers_range_dictionary["normalize"] = [True]

    elif similarity_type == "tversky":
        hyperparamethers_range_dictionary["tversky_alpha"] = range(0, 2)
        hyperparamethers_range_dictionary["tversky_beta"] = range(0, 2)
        hyperparamethers_range_dictionary["normalize"] = [True]

    if similarity_type in ["cosine", "asymmetric"]:
        hyperparamethers_range_dictionary["feature_weighting"] = ["none", "BM25", "TF-IDF"]

    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [ICM_train, URM_train],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    output_root_path_similarity = output_root_path + "_" + similarity_type

    best_parameters = parameterSearch.search(recommenderDictionary,
                                             n_cases=n_cases,
                                             output_root_path=output_root_path_similarity,
                                             metric=metric_to_optimize)


def runParameterSearch_Content(recommender_class, URM_train, ICM_object, ICM_name, n_cases=30,
                               evaluator_validation=None, evaluator_test=None, metric_to_optimize="PRECISION",
                               output_root_path="result_experiments/", parallelizeKNN=False):
    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    ##########################################################################################################

    this_output_root_path = output_root_path + recommender_class.RECOMMENDER_NAME + "_{}".format(ICM_name)

    parameterSearch = BayesianSearch(recommender_class, evaluator_validation=evaluator_validation,
                                     evaluator_test=evaluator_test)

    similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]

    run_KNNCBFRecommender_on_similarity_type_partial = partial(run_KNNCBFRecommender_on_similarity_type,
                                                               parameterSearch=parameterSearch,
                                                               URM_train=URM_train,
                                                               ICM_train=ICM_object,
                                                               n_cases=n_cases,
                                                               output_root_path=this_output_root_path,
                                                               metric_to_optimize=metric_to_optimize)

    if parallelizeKNN:
        pool = PoolWithSubprocess(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
        resultList = pool.map(run_KNNCBFRecommender_on_similarity_type_partial, similarity_type_list)

    else:

        for similarity_type in similarity_type_list:
            run_KNNCBFRecommender_on_similarity_type_partial(similarity_type)


def runParameterSearch_Collaborative(recommender_class, URM_train, ICM_1, ICM_2,UCM_1,UCM_2,  metric_to_optimize="PRECISION",
                                     evaluator_validation=None, evaluator_test=None,
                                     evaluator_validation_earlystopping=None,
                                     output_root_path="result_experiments/", parallelizeKNN=True, n_cases=200, reader=None):
    from ParameterTuning.AbstractClassSearch import DictionaryKeys

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    try:

        output_root_path_rec_name = output_root_path + recommender_class.RECOMMENDER_NAME

        parameterSearch = BayesianSearch(recommender_class, evaluator_validation=evaluator_validation, evaluator_test = evaluator_test)

        if recommender_class in [TopPop, Random]:
            recommender = recommender_class(URM_train)

            recommender.fit()

            output_file = open(output_root_path_rec_name + "_BayesianSearch.txt", "a")
            result_dict, result_baseline = evaluator_validation.evaluateRecommender(recommender)
            output_file.write(
                "ParameterSearch: Best result evaluated on URM_validation. Results: {}".format(result_baseline))

            pickle.dump(result_dict.copy(),
                        open(output_root_path_rec_name + "_best_result_validation", "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)

            result_dict, result_baseline = evaluator_test.evaluateRecommender(recommender)
            output_file.write("ParameterSearch: Best result evaluated on URM_test. Results: {}".format(result_baseline))

            pickle.dump(result_dict.copy(),
                        open(output_root_path_rec_name + "_best_result_test", "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)

            output_file.close()

            return

        ##########################################################################################################

        if recommender_class is UserKNNCFRecommender:

            similarity_type_list = ['cosine']

            run_KNNCFRecommender_on_similarity_type_partial = partial(run_KNNCFRecommender_on_similarity_type,
                                                                      parameterSearch=parameterSearch,
                                                                      URM_train=URM_train,
                                                                      n_cases=n_cases,
                                                                      output_root_path=output_root_path_rec_name,
                                                                      metric_to_optimize=metric_to_optimize)

            if parallelizeKNN:
                pool = PoolWithSubprocess(processes=int(2), maxtasksperchild=1)
                resultList = pool.map(run_KNNCFRecommender_on_similarity_type_partial, similarity_type_list)

            else:

                for similarity_type in similarity_type_list:
                    run_KNNCFRecommender_on_similarity_type_partial(similarity_type)

            return

        ##########################################################################################################

        if recommender_class is ItemKNNCFRecommender:

            similarity_type_list = ['cosine']


            run_KNNCFRecommender_on_similarity_type_partial = partial(run_KNNCFRecommender_on_similarity_type,
                                                                      parameterSearch=parameterSearch,
                                                                      URM_train=URM_train,
                                                                      n_cases=n_cases,
                                                                      output_root_path=output_root_path_rec_name,
                                                                      metric_to_optimize=metric_to_optimize)

            if parallelizeKNN:
                pool = PoolWithSubprocess(processes=int(2), maxtasksperchild=1)
                resultList = pool.map(run_KNNCFRecommender_on_similarity_type_partial, similarity_type_list)

            else:

                for similarity_type in similarity_type_list:
                    run_KNNCFRecommender_on_similarity_type_partial(similarity_type)

            return

        ##########################################################################################################

        # if recommender_class is MultiThreadSLIM_RMSE:
        #
        #     hyperparamethers_range_dictionary = {}
        #     hyperparamethers_range_dictionary["topK"] = [50, 100]
        #     hyperparamethers_range_dictionary["l1_penalty"] = [1e-2, 1e-3, 1e-4]
        #     hyperparamethers_range_dictionary["l2_penalty"] = [1e-2, 1e-3, 1e-4]
        #
        #
        #     recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
        #                              DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
        #                              DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
        #                              DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
        #                              DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
        #
        #

        ##########################################################################################################

        if recommender_class is P3alphaRecommender:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
            hyperparamethers_range_dictionary["alpha"] = range(0, 2)
            hyperparamethers_range_dictionary["normalize_similarity"] = [True, False]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################

        if recommender_class is HybridRecommender:




            hyperparamethers_range_dictionary = {}
            # hyperparamethers_range_dictionary["w_itemcf"] = [(x * 0.05) for x in range(0, 40)]
            # hyperparamethers_range_dictionary["w_usercf"] = [(x * 0.05) for x in range(0, 40)]
            # hyperparamethers_range_dictionary["w_cbart"] = [(x * 0.05) for x in range(0, 40)]
            # hyperparamethers_range_dictionary["w_cbalb"] = [(x * 0.05) for x in range(0, 40)]
            # hyperparamethers_range_dictionary["w_slim"] = [(x * 0.05) for x in range(0, 40)]
            # hyperparamethers_range_dictionary["w_svd"] = [x * 0.05 for x in range(0, 40)]
            # hyperparamethers_range_dictionary["w_rp3"] = [(x * 0.05)  for x in range(0, 40)]

            hyperparamethers_range_dictionary["w_itemcf"] = [(x * 0.05)+1  for x in range(0, 20)]
            hyperparamethers_range_dictionary["w_usercf"] = [(x * 0.05) for x in range(0, 20)]
            hyperparamethers_range_dictionary["w_cbart"] = [(x * 0.05) for x in range(0, 20)]
            hyperparamethers_range_dictionary["w_cbalb"] = [(x * 0.05) for x in range(0, 20)]
            hyperparamethers_range_dictionary["w_slim"] = [(x * 0.05)  for x in range(0, 20)]
            hyperparamethers_range_dictionary["w_svd"] = [(x * 0.05) for x in range(0, 20)]
            hyperparamethers_range_dictionary["w_rp3"] = [(x * 0.05) for x in range(0, 20)]
            #hyperparamethers_range_dictionary["w_cbuserart"] = [(x * 0.05) for x in range(0, 20)]
            #hyperparamethers_range_dictionary["w_cbuseralb"] = [(x * 0.05) for x in range(0, 20)]
            #hyperparamethers_range_dictionary["w_slimnot"] = [(x * 0.05) for x in range(0, 20)]
            #hyperparamethers_range_dictionary["w_mf"] = [(x * 0.05) for x in range(0, 20)]

            #hyperparamethers_range_dictionary["topPop"] = [x for x in range(1, 10)]
            #hyperparamethers_range_dictionary["idf"] = [True, False]
            #hyperparamethers_range_dictionary["w_p3_alpha"] = [(x * 0.05) for x in range(0, 20)]
            # hyperparamethers_range_dictionary["topKItem"] = [(x * 50)+50 for x in range(0, 50)]
            # hyperparamethers_range_dictionary["topKUser"] = [(x * 50)+50 for x in range(0, 50)]
            # hyperparamethers_range_dictionary["shrinkItem"] = [x * 0.5 for x in range(0, 50)]
            # hyperparamethers_range_dictionary["shrinkUser"] = [x * 0.5 for x in range(0, 50)]
            #hyperparamethers_range_dictionary["topKP3"] = [(x * 50)+50 for x in range(0, 50)]
            # hyperparamethers_range_dictionary["alphaP3"] = [(x * 0.0005) for x in range(0, 50)]
            #hyperparamethers_range_dictionary["topKRP3"] = [(x * 50)+50 for x in range(0, 50)]
            #hyperparamethers_range_dictionary["alphaRP3"] = range(0, 1)
            #hyperparamethers_range_dictionary["betaRP3"] = range(0, 1)
            #hyperparamethers_range_dictionary["factorsSVD"] = [(x * 10)+10 for x in range(0, 100)]
            # hyperparamethers_range_dictionary["topKCBAlb"] = [(x * 50)+50 for x in range(0, 50)]
            # hyperparamethers_range_dictionary["topKCBArt"] = [(x * 50)+50 for x in range(0, 50)]
            # hyperparamethers_range_dictionary["shrinkCBAlb"] = [x * 0.5 for x in range(0, 50)]
            # hyperparamethers_range_dictionary["shrinkCBArt"] = [x * 0.5 for x in range(0, 50)]
            #hyperparamethers_range_dictionary["pen_offset"] = [(x * 0.01) for x in range(1, 7)]


            # idf = True
            #
            # if (idf):
            #
            #     nItems = URM_train.shape[1]
            #     URMidf = sps.lil_matrix((URM_train.shape[0], URM_train.shape[1]))
            #
            #     for i in range(0, URM_train.shape[0]):
            #         IDF_i = np.log(nItems / np.sum(URM_train[i]))
            #         URMidf[i] = np.multiply(URM_train[i], IDF_i)
            #
            #     URM_train = URMidf.tocsr()



            # hyperparamethers_range_dictionary["cb_weight"] = [x * 2 for x in range(0, 50)]
            # #hyperparamethers_range_dictionary["cf_weight"] = [x * 2 for x in range(0, 50)]
            # hyperparamethers_range_dictionary["graph_weight"] = [x * 2 for x in range(0, 50)]
            # hyperparamethers_range_dictionary["hybrid_weight"] = [x * 2 for x in range(0, 50)]
            # hyperparamethers_range_dictionary["final_weight"] = [x * 2 for x in range(0, 50)]

            #hyperparamethers_range_dictionary["w_p3_alpha"] = [x * 0.05 for x in range(0, 20)]

            item = ItemKNNCFRecommender(URM_train)

            user = UserKNNCFRecommender(URM_train)

            SLIM = SLIMElasticNetRecommender(URM_train=URM_train)

            p3_alpha = P3alphaRecommender(URM_train=URM_train)


            #SLIMnot = SLIM_BPR_Cython(URM_train=URM_train, train_with_sparse_weights=True, symmetric=True, positive_threshold=1 )

            #SLIMnot.fit(validation_every_n=5, stop_on_validation=True, evaluator_object=evaluator_validation_earlystopping, lower_validatons_allowed=3, validation_metric=metric_to_optimize,  )

            MF = MatrixFactorization_BPR_Cython(URM_train=URM_train, positive_threshold=1)

            #MF.fit(validation_every_n= 5,stop_on_validation= True, evaluator_object= evaluator_validation_earlystopping,lower_validatons_allowed= 20, validation_metric = metric_to_optimize)

            #simURM_ICF = URM_train.dot(simURM_ICF)


            #simURM_UCF = simURM_UCF.dot(URM_train)

            SLIM.fit(l1_penalty=1.95e-06, l2_penalty=0, positive_only=True, topK=1500, alpha=0.00165)
            #simURM_SLIM = URM_train.dot(simURM_SLIM)

            #"SLIM": simURM_SLIM,
            ## "p3_alpha": p3_alpha,

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"ICM_Art": ICM_1,
                                                                       "ICM_Alb": ICM_2,
                                                                       "item": item,
                                                                       "user": user,
                                                                       "SLIM": SLIM,




                                                                       "users" : reader.get_users(),
                                                                       "target" : reader.get_target_list(),











                                                                       },
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}



        ##########################################################################################################

        if recommender_class is RP3betaRecommender:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
            hyperparamethers_range_dictionary["alpha"] = range(0, 2)
            hyperparamethers_range_dictionary["beta"] = range(0, 2)
            hyperparamethers_range_dictionary["normalize_similarity"] = [True]
            hyperparamethers_range_dictionary["implicit"] = [True]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################

        if recommender_class is MatrixFactorization_FunkSVD_Cython:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "adam"]
            # hyperparamethers_range_dictionary["epochs"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
            hyperparamethers_range_dictionary["num_factors"] = range(100, 1000, 20)
            hyperparamethers_range_dictionary["reg"] = [0.0, 1e-3, 1e-6, 1e-9]
            hyperparamethers_range_dictionary["learning_rate"] = [1e-2, 1e-3, 1e-4, 1e-5]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"validation_every_n": 5,
                                                                       "stop_on_validation": True,
                                                                       "evaluator_object": evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed": 20,
                                                                       "validation_metric": metric_to_optimize},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

            ##########################################################################################################

        # if recommender_class is MF_MSE_PyTorch:
        #     hyperparamethers_range_dictionary = {}
        #     #hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "adam"]
        #     # hyperparamethers_range_dictionary["epochs"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
        #     hyperparamethers_range_dictionary["num_factors"] = range(100, 1000, 20)
        #     #hyperparamethers_range_dictionary["reg"] = [0.0, 1e-3, 1e-6, 1e-9]
        #     hyperparamethers_range_dictionary["learning_rate"] = [1e-2, 1e-3, 1e-4, 1e-5]
        #
        #     recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
        #                              DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
        #                              DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
        #                              DictionaryKeys.FIT_KEYWORD_ARGS: {"validation_every_n": 5,
        #
        #                                                                "evaluator_object": evaluator_validation_earlystopping,
        #                                                                "lower_validatons_allowed": 20,
        #                                                                "validation_metric": metric_to_optimize},
        #                              DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}



            ##########################################################################################################

        if recommender_class is FunkSVD:
            hyperparamethers_range_dictionary = {}

            # hyperparamethers_range_dictionary["epochs"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
            hyperparamethers_range_dictionary["num_factors"] = [x for x in range(100, 2000, 200)]
            hyperparamethers_range_dictionary["reg"] = [0.0, 1e-03, 1e-06, 1e-09]
            hyperparamethers_range_dictionary["learning_rate"] = [1e-02, 1e-03]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################

        if recommender_class is MatrixFactorization_BPR_Cython:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad"]
            # hyperparamethers_range_dictionary["epochs"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
            hyperparamethers_range_dictionary["num_factors"] = [400, 500, 600, 700, 1000, 1500, 2000]
            hyperparamethers_range_dictionary["batch_size"] = [1000, 1200, 1500, 2000, 2500, 3000, 4000, 5000, 6000]
            hyperparamethers_range_dictionary["positive_reg"] = [x*0.00000005 for x in range(0, 40)]
            hyperparamethers_range_dictionary["negative_reg"] = [x*0.00000005 for x in range(0, 40)]
            hyperparamethers_range_dictionary["learning_rate"] = [1e-2, 1e-3, 0.1]
            hyperparamethers_range_dictionary["user_reg"] = [x*0.00000005 for x in range(0, 40)]


            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'positive_threshold': 1},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"validation_every_n": 5,
                                                                       "stop_on_validation": True,
                                                                       "evaluator_object": evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed": 20,
                                                                       "validation_metric": metric_to_optimize},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################

        if recommender_class is PureSVDRecommender:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["num_factors"] = list(range(0, 250, 5))

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        #########################################################################################################

        if recommender_class is SLIM_BPR_Cython:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = [800, 900, 1000, 1200]
            # hyperparamethers_range_dictionary["epochs"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
            hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad"]
            hyperparamethers_range_dictionary["lambda_i"] = [1e-6]
            hyperparamethers_range_dictionary["lambda_j"] = [1e-9]
            hyperparamethers_range_dictionary["learning_rate"] = [0.01, 0.001, 1e-4, 1e-5, 0.1]
            hyperparamethers_range_dictionary["batch_size"] = [x*100 for x in range(1, 10)]
            hyperparamethers_range_dictionary["topK"] = [x * 50 for x in range(1, 20)]
            hyperparamethers_range_dictionary["gamma"] = [(x * 0.02) + 0.9 for x in range(0, 5)]
            hyperparamethers_range_dictionary["beta_1"] = [(x * 0.02) + 0.9 for x in range(0, 5)]
            hyperparamethers_range_dictionary["beta_2"] = [(x * 0.02) + 0.9 for x in range(0, 5)]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'train_with_sparse_weights': True,
                                                                               'symmetric': True,
                                                                               'positive_threshold': 1},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"validation_every_n": 5,
                                                                       "stop_on_validation": True,
                                                                       "evaluator_object": evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed":3,
                                                                       "validation_metric": metric_to_optimize},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################

        if recommender_class is SLIMElasticNetRecommender:
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = [900, 1000, 1100, 1200, 1300, 1400, 1500, 1700, 2000, 2200, 2500, 3000]
            hyperparamethers_range_dictionary["l1_penalty"] = [x*0.00000005 for x in range(0, 40)]
            hyperparamethers_range_dictionary["l2_penalty"] = [1e-4]
            hyperparamethers_range_dictionary["alpha"] = [x*0.00005 for x in range(0, 60)]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        #########################################################################################################

        ## Final step, after the hyperparameter range has been defined for each type of algorithm
        best_parameters = parameterSearch.search(recommenderDictionary,
                                                 n_cases=n_cases,
                                                 output_root_path=output_root_path_rec_name,
                                                 metric=metric_to_optimize)




    except Exception as e:

        print("On recommender {} Exception {}".format(recommender_class, str(e)))
        traceback.print_exc()

        error_file = open(output_root_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(recommender_class, str(e)))
        error_file.close()


import os, multiprocessing
from functools import partial

#from data.Movielens_10M.Movielens10MReader import Movielens10MReader




def read_data_split_and_search():
    """
    This function provides a simple example on how to tune parameters of a given algorithm

    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """

    data = dataReader()

    URM_train = data.get_URM_train()
    URM_validation = data.get_URM_validation()
    URM_test = data.get_URM_test()

    ICM_alb = data.get_ICM_Alb()
    ICM_art = data.get_ICM_Art()
    UCM_alb = data.get_UCM_Alb()
    UCM_art = data.get_UCM_Art()


    output_root_path = "result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    collaborative_algorithm_list = [
       HybridRecommender

    ]

    from ParameterTuning.AbstractClassSearch import EvaluatorWrapper
    from Base.Evaluation.Evaluator import SequentialEvaluator

    evaluator_validation_earlystopping = SequentialEvaluator(URM_validation, cutoff_list=[10])
    evaluator_test = SequentialEvaluator(URM_test, cutoff_list=[10])

    evaluator_validation = EvaluatorWrapper(evaluator_validation_earlystopping)
    evaluator_test = EvaluatorWrapper(evaluator_test)

    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                       URM_train=URM_train,
                                                       ICM_1 = ICM_art,
                                                       ICM_2 = ICM_alb,
                                                       UCM_1 = UCM_art,
                                                       UCM_2 = UCM_alb,
                                                       metric_to_optimize="MAP",
                                                       evaluator_validation_earlystopping=evaluator_validation_earlystopping,
                                                       evaluator_validation=evaluator_validation,
                                                       evaluator_test=evaluator_test,
                                                       output_root_path=output_root_path,
                                                       reader=data)

    # runParameterSearch_Collaborative_partial = partial(runParameterSearch_Content,
    #                                                    URM_train=URM_train,
    #
    #                                                    ICM_object=UCM_alb,
    #                                                    ICM_name="UCM_alb",
    #                                                    metric_to_optimize="MAP",
    #
    #                                                    evaluator_validation=evaluator_validation,
    #                                                    evaluator_test=evaluator_test,
    #                                                    output_root_path=output_root_path,
    #                                                    )

    # pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    # resultList = pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)

    for recommender_class in collaborative_algorithm_list:

        try:

            runParameterSearch_Collaborative_partial(recommender_class)

        except Exception as e:

            print("On recommender {} Exception {}".format(recommender_class, str(e)))
            traceback.print_exc()


if __name__ == '__main__':
    read_data_split_and_search()
