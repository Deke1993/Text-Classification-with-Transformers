#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This metric computes the percentage ofcorrectly rejected examples, which is the percentage 
#of incorrect predictionsamong all the abstained examples.
from metrics import AbstainPrecision

 #Different from `AbstainPrecision`, `AbstainRecall` computes the percentage of
  #correctly abstained examples among all the incorrect predictions that **could
  #have been abstained**. 

from metrics import AbstainRecall

from Refactoring.model import load_model, build_classifier_model, build_classifier_model_last4

import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import os
from collections import defaultdict

#uncertainty
from robustness_metrics.metrics import uncertainty
#loggging
from datetime import datetime

from Refactoring.helper_functions import scale, invert, map_logits

#metrics adapated for TopKClassification
from metrics_topk import AbstainPrecision_Topk
'''
 Different from `AbstainPrecision`, `AbstainRecall` computes the percentage of
  correctly abstained examples among all the incorrect predictions that **could
  have been abstained**. 
'''
from metrics_topk import AbstainRecall_Topk
from metrics_topk import CalibrationAUC_Topk


# In[ ]:


def create_Vanilla_pred(model_path, test_x, model, ood, zero_shot, label_dict_train=None, label_dict_test=None):
    model.load_weights('models/fit/'+model_path)
    
    predictions_logit = model.predict(x={'input_ids': test_x['input_ids'],'attention_mask':test_x["attention_mask"]})
    predictions_probs = np.array(tf.nn.softmax(predictions_logit, axis=-1))
    
    if ood or zero_shot:
        predictions_probs = map_logits(label_dict_train, label_dict_test, predictions_probs)
    
    return predictions_probs


# In[ ]:


def mc_dropout_sampling(test_x,model):
  # Enable dropout during inference.
  return model.predict(x={'input_ids': test_x['input_ids'],'attention_mask':test_x["attention_mask"]})


# In[ ]:


def mc_predictions(num_ensemble, model_path, test_x, model, ood,zero_shot, label_dict_train=None, label_dict_test=None):
    model.load_weights('models/fit/'+model_path)

    # Monte Carlo dropout inference.
    dropout_logit_samples = [mc_dropout_sampling(test_x, model) for _ in range(num_ensemble)]
    dropout_probs_samples = [tf.nn.softmax(dropout_logits, axis=-1) for dropout_logits in dropout_logit_samples]
    dropout_probs_samples = tf.stack([dropout_probs_samples])[0]
    dropout_probs_mean = tf.reduce_mean(dropout_probs_samples, axis=0)
    dropout_probs_var = tf.math.reduce_variance(dropout_probs_samples, axis=0)
    
    if ood or zero_shot:
        dropout_probs_mean = np.array(dropout_probs_mean)
        dropout_probs_var = np.array(dropout_probs_var)
        
        dropout_probs_mean = map_logits(label_dict_train, label_dict_test, dropout_probs_mean)
        dropout_probs_var = map_logits(label_dict_train, label_dict_test, dropout_probs_var)
    
    return dropout_probs_samples, dropout_probs_mean, dropout_probs_var


# In[ ]:


def save_single_model_predictions(test_x,tokenizer, transformer_model, max_length,bert_type, reduce_strategy, layer_strategy, last_layer_strategy, config, n_classes,language,language_model_relation,cased,learning_rate,seed,decay_factor,num_ensemble, local_path="/media/remote/jdeke/models/fit/*.index"):
    model_logit_samples = []

    for f in glob.glob(local_path):
        print(os.path.basename(f)[:-6])
        if layer_strategy == 'last':
            model = build_classifier_model(seed=None, tokenizer=tokenizer, transformer_model=transformer_model, max_length=max_length,bert_type=bert_type, reduce_strategy=reduce_strategy, config=config, n_classes=n_classes)
        elif layer_strategy == 'last4':
            model = build_classifier_model_last4(seed=None, tokenizer=tokenizer, transformer_model=transformer_model, max_length=max_length,bert_type=bert_type, reduce_strategy=reduce_strategy, last_layer_strategy=last_layer_strategy, config=config, n_classes=n_classes)

        model.load_weights('models/fit/'+os.path.basename(f)[:-6])
        model_logit_samples.append(model.predict(x={'input_ids': test_x['input_ids'],'attention_mask':test_x["attention_mask"]}))
        
    np.save(str(language)+"_"+str(layer_strategy) + "_"+str(last_layer_strategy)+"_"+ str(reduce_strategy) + "_"+str(decay_factor) +"_" + str(language_model_relation) + "_" + str(cased)+ "_" + str(bert_type) +"_" + str(learning_rate) +"_" +datetime.now().strftime("%Y%m%d-%H%M%S")+ "_random_state" + str(seed) +"_technique_" +"For Deep Ensemble"+"_ensembles_"+str(num_ensemble), np.array(model_logit_samples))


# In[ ]:


def create_deep_ensemble_predictions(load_path,num_ensemble, ood,zero_shot, label_dict_train=None, label_dict_test=None):
    model_logit_samples = np.load(load_path)

    # Deep ensemble inference
    #model_logit_samples = [model.predict(x={'input_ids': test_x['input_ids'],'attention_mask':test_x["attention_mask"]}) for model in model_ensemble]
    model_prob_samples = [tf.nn.softmax(logits, axis=-1) for logits in model_logit_samples[:num_ensemble]]
    model_prob_samples = tf.stack([model_prob_samples])[0]
    model_probs_mean = tf.reduce_mean(model_prob_samples, axis=0)
    model_probs_var = tf.math.reduce_variance(model_prob_samples, axis=0)
    
    if ood or zero_shot:
        model_probs_mean = np.array(model_probs_mean)
        model_probs_var = np.array(model_probs_var)
        
        model_probs_mean = map_logits(label_dict_train, label_dict_test, model_probs_mean)
        model_probs_var = map_logits(label_dict_train, label_dict_test, model_probs_var)
    
    return model_prob_samples, model_probs_mean, model_probs_var


# In[ ]:


#adapted from https://github.com/tanyinghui/DBALwithImgData/blob/84877495b5d0ca0c60e8d3822e683b6a207f28ed/acquisition_functions.py
def bald(samples):
    pc = tf.reduce_mean(samples,axis=0)
    pc = np.array(pc)
    H = (-pc * np.log(pc + 1e-10)).sum(axis=-1)
    E_H = -np.mean(np.sum(samples * np.log(samples + 1e-10), axis=-1), axis=0)
    
    return H-E_H


# In[ ]:


#original implementation from brier according to wikipedia
def brier_multi(targets, probs):
    return np.mean(np.sum((probs - targets)**2, axis=1))


# In[ ]:


#confidence: shape (batch_size,classes), the higher the more confident
def abstain_accuracy(y_true, y_pred, confidence, fraction, confidence_ordering="mean"):
    #the higher the confidence score, the higher the confidence
    if confidence_ordering=="mean":
        #get the max confidence of each testsample
        max_values = np.max(confidence,axis=1)

        #get index of the max confidence for each testsample
        max_indices = np.argsort(max_values)


        #compute slicing fraction
        low_conf_indices_slice = int(fraction*len(max_indices))
        
        if (fraction == 0 or fraction == 1):
            indices = max_indices
        else:
            #select indices according to slice
            indices = max_indices[low_conf_indices_slice:]


    
    #the lower the confidence score, the higher the confidence --> variance
    elif confidence_ordering=="var":

        #get the mean variance per sample, see how certain is your transformer
        mean_values = np.mean(confidence,axis=1)

        #get index of the max confidence for each testsample
        max_indices = np.argsort(mean_values)

        #compute slicing fraction
        low_conf_indices_slice = int(fraction*len(max_indices))

        
        if (fraction == 0 or fraction == 1):
            indices = max_indices
        else:
            #select indices according to slice
            indices = max_indices[:-low_conf_indices_slice]

        #print(indices)
    elif confidence_ordering=="BALD":
        #get index of the max confidence for each testsample
        max_indices = np.argsort(confidence)

        #compute slicing fraction
        low_conf_indices_slice = int(fraction*len(max_indices))
        
       
        if (fraction == 0 or fraction == 1):
            indices = max_indices
        else:
            #select indices according to slice
            indices = max_indices[:-low_conf_indices_slice]
        
        #print(indices)
        
    elif confidence_ordering=="random":
        #get index of the max confidence for each testsample
        max_indices = np.argsort(confidence)


        #compute slicing fraction
        low_conf_indices_slice = int(fraction*len(max_indices))
        
        if (fraction == 0 or fraction == 1):
            indices = max_indices
        else:
            #select indices according to slice
            indices = max_indices[low_conf_indices_slice:]
        

    elif confidence_ordering=="class":

        #get index of the max confidence for each testsample
        max_indices = np.argsort(confidence)


        #compute slicing fraction
        low_conf_indices_slice = int(fraction*len(max_indices))
        
        if (fraction == 0 or fraction == 1):
            indices = max_indices
        else:
            #select indices according to slice
            indices = max_indices[low_conf_indices_slice:]
        

    #confidence sliced
    confidence_sliced = np.take(confidence, indices, 0)
    
    #predictions sliced
    predictions_sliced = y_pred[indices]
    
    
    y_true_sliced = y_true[indices]
    

    
    if fraction != 1:
    
        acc = np.mean(predictions_sliced==y_true_sliced)
    
    else:
        acc=1
    
    combined_accuracy = fraction*1+(1-fraction)*acc
    
    return acc, combined_accuracy


# In[ ]:


def compute_metrics(method,y_test,y_test_categorical,dropout_probs_samples,dropout_probs_mean, dropout_probs_var, num_ensemble, language, layer_strategy, last_layer_strategy,reduce_strategy,decay_factor,language_model_relation,cased,bert_type,learning_rate,seed, model_path):
    d = defaultdict(dict)
    c = defaultdict(dict)
    b = defaultdict(dict)
    a = defaultdict(dict)
    e = defaultdict(dict)
    
    metrics_dict = {"ECE":None, "BRIER":None, "CalibAUROC":d, "CalibAUPR":c, "AbstainAccuracy":b, "AbstainPrecision":a,"AbstainRecall":e}
    

    metric = uncertainty._KerasECEMetric()
    metric.update_state(y_test,dropout_probs_mean)
    metrics_dict["ECE"]=metric.result().numpy()


    metrics_dict["BRIER"] = brier_multi(y_test_categorical,dropout_probs_mean)
    
    #bald computation
    if method != "Vanilla":
        B = bald(dropout_probs_samples)

        bald_norm = scale(B)
        var_norm = scale(np.mean(dropout_probs_var,axis=1))

        bald_norm_inv = invert(bald_norm)
        var_norm_inv = invert(var_norm)
    
    #class ranking for class baseline
    #get index of the max confidence for each testsample
    max_indices = np.argmax(dropout_probs_mean,axis=1)

    unique, counts = np.unique(max_indices, return_counts=True)

    lookup_dict = dict(zip(unique,counts))


    support_ranking_list = []
    for i in max_indices:
        if i in lookup_dict:
            support_ranking_list.append(lookup_dict[i])
        else:
            support_ranking_list.append(0)

    support_ranking = scale(np.array(support_ranking_list))
    
    random_ranking = np.random.rand(np.shape(y_test)[0])
    
    y_pred = np.argmax(dropout_probs_mean, axis=1) 
    confidence =  np.max(dropout_probs_mean, axis=1)
    
    ## ROC
    
    
    if method != "Vanilla":
        confidence_scores = [confidence, var_norm_inv, bald_norm_inv, support_ranking, random_ranking]
        for i,j in enumerate(["Softmax","Variance","BALD","Class","Random"]):
            metric = uncertainty._KerasCalibrationAUCMetric()
            metric.update_state(y_test,y_pred,confidence_scores[i])
            metrics_dict["CalibAUROC"][j]=metric.result().numpy()

        for i,j in enumerate(["Softmax","Variance","BALD","Class","Random"]):
            metric = uncertainty._KerasCalibrationAUCMetric(curve="PR")
            metric.update_state(y_test,y_pred,confidence_scores[i])
            metrics_dict["CalibAUPR"][j]=metric.result().numpy()
            
    
    elif method == "Vanilla":
        confidence_scores = [confidence, support_ranking, random_ranking]
        for i,j in enumerate(["Softmax","Class","Random"]):
            metric = uncertainty._KerasCalibrationAUCMetric()
            metric.update_state(y_test,y_pred,confidence_scores[i])
            metrics_dict["CalibAUROC"][j]=metric.result().numpy()

        for i,j in enumerate(["Softmax","Class","Random"]):
            metric = uncertainty._KerasCalibrationAUCMetric(curve="PR")
            metric.update_state(y_test,y_pred,confidence_scores[i])
            metrics_dict["CalibAUPR"][j]=metric.result().numpy()

    fractions = np.arange(0, 1.001, 0.005)
    
    #dropout_probs_mean
    
    for i in fractions:
        metrics_dict["AbstainAccuracy"][i]["Softmax"] = abstain_accuracy(y_test,y_pred,dropout_probs_mean, i ,confidence_ordering="mean")
        metrics_dict["AbstainAccuracy"][i]["Class"] = abstain_accuracy(y_test,y_pred,support_ranking, i,confidence_ordering="class")
        metrics_dict["AbstainAccuracy"][i]["Random"] = abstain_accuracy(y_test,y_pred,random_ranking, i,confidence_ordering="random")
        if method != "Vanilla":
            metrics_dict["AbstainAccuracy"][i]["Variance"] = abstain_accuracy(y_test,y_pred,dropout_probs_var, i ,confidence_ordering="var")
            metrics_dict["AbstainAccuracy"][i]["BALD"] = abstain_accuracy(y_test,y_pred,B, i,confidence_ordering="BALD")




        metric = AbstainPrecision(abstain_fraction= i)
        metric.update_state(y_test,y_pred,confidence)
        metrics_dict["AbstainPrecision"][i]["Softmax"] = metric.result().numpy()

        metric = AbstainPrecision(abstain_fraction= i)
        metric.update_state(y_test,y_pred,support_ranking)
        metrics_dict["AbstainPrecision"][i]["Class"] = metric.result().numpy()

        metric = AbstainPrecision(abstain_fraction= i)
        metric.update_state(y_test,y_pred,random_ranking)
        metrics_dict["AbstainPrecision"][i]["Random"] = metric.result().numpy()
        
        if method != "Vanilla":
            metric = AbstainPrecision(abstain_fraction= i)
            metric.update_state(y_test,y_pred,var_norm_inv)
            metrics_dict["AbstainPrecision"][i]["Variance"] = metric.result().numpy()

            metric = AbstainPrecision(abstain_fraction= i)
            metric.update_state(y_test,y_pred,bald_norm_inv)
            metrics_dict["AbstainPrecision"][i]["BALD"] = metric.result().numpy()




        metric = AbstainRecall(abstain_fraction= i)
        metric.update_state(y_test,y_pred,confidence)
        metrics_dict["AbstainRecall"][i]["Softmax"] = metric.result().numpy()

        metric = AbstainPrecision(abstain_fraction= i)
        metric.update_state(y_test,y_pred,support_ranking)
        metrics_dict["AbstainRecall"][i]["Class"] = metric.result().numpy()

        metric = AbstainPrecision(abstain_fraction= i)
        metric.update_state(y_test,y_pred,random_ranking)
        metrics_dict["AbstainRecall"][i]["Random"] = metric.result().numpy()
        if method != "Vanilla":
            metric = AbstainRecall(abstain_fraction= i)
            metric.update_state(y_test,y_pred,var_norm_inv)
            metrics_dict["AbstainRecall"][i]["Variance"] = metric.result().numpy()

            metric = AbstainRecall(abstain_fraction= i)
            metric.update_state(y_test,y_pred,bald_norm_inv)
            metrics_dict["AbstainRecall"][i]["BALD"] = metric.result().numpy()

        
    #metrics_dict

    df = pd.DataFrame.from_dict(metrics_dict)

    #df

    df["CalibAUROC Softmax"] = df["CalibAUROC"].iloc[0]
    if method != "Vanilla":
        df["CalibAUROC Variance"] = df["CalibAUROC"].iloc[1]
        df["CalibAUROC BALD"] = df["CalibAUROC"].iloc[2]
    df["CalibAUROC Class"] = df["CalibAUROC"].iloc[3]
    df["CalibAUROC Random"] = df["CalibAUROC"].iloc[4]

    df["CalibAUPR Softmax"] = df["CalibAUPR"].iloc[0]
    if method != "Vanilla":
        df["CalibAUPR Variance"] = df["CalibAUPR"].iloc[1]
        df["CalibAUPR BALD"] = df["CalibAUPR"].iloc[2]
    df["CalibAUPR Class"] = df["CalibAUPR"].iloc[3]
    df["CalibAUPR Random"] = df["CalibAUPR"].iloc[4]

    #df

    df.drop(columns=["CalibAUROC","CalibAUPR"],inplace=True)

    df.ffill(inplace=True)


    if method != "Vanilla":
        df = df.iloc[5:]
        df[["AbstainAccuracy_Softmax","AbstainAccuracy_Class","AbstainAccuracy_Random","AbstainAccuracy_Variance","AbstainAccuracy_BALD"]] = df["AbstainAccuracy"].apply(pd.Series)
        for i in ["AbstainAccuracy_Softmax","AbstainAccuracy_Class","AbstainAccuracy_Random","AbstainAccuracy_Variance","AbstainAccuracy_BALD"]:
            df[[i+"_Acc", i+"_CombinedAcc"]] = pd.DataFrame(df[i].tolist(), index=df.index)
        df[["AbstainPrecision_Softmax","AbstainPrecision_Class","AbstainPrecision_Random","AbstainPrecision_Variance","AbstainPrecision_BALD"]] = df["AbstainPrecision"].apply(pd.Series)
        df[["AbstainRecall_Softmax","AbstainRecall_Class","AbstainRecall_Random","AbstainRecall_Variance","AbstainRecall_BALD"]] = df["AbstainRecall"].apply(pd.Series)
        df.drop(columns=["AbstainAccuracy","AbstainPrecision","AbstainRecall","AbstainAccuracy_Softmax","AbstainAccuracy_Class","AbstainAccuracy_Random","AbstainAccuracy_Variance","AbstainAccuracy_BALD"], inplace=True)#"AbstainAccuracy_Variance","AbstainAccuracy_BALD"
    elif method == "Vanilla":
        df = df.iloc[3:]
        df[["AbstainAccuracy_Softmax","AbstainAccuracy_Class","AbstainAccuracy_Random"]] = df["AbstainAccuracy"].apply(pd.Series)
        for i in ["AbstainAccuracy_Softmax","AbstainAccuracy_Class","AbstainAccuracy_Random"]:
            df[[i+"_Acc", i+"_CombinedAcc"]] = pd.DataFrame(df[i].tolist(), index=df.index)
        df[["AbstainPrecision_Softmax","AbstainPrecision_Class","AbstainPrecision_Random"]] = df["AbstainPrecision"].apply(pd.Series)
        df[["AbstainRecall_Softmax","AbstainRecall_Class","AbstainRecall_Random"]] = df["AbstainRecall"].apply(pd.Series)
        df.drop(columns=["AbstainAccuracy","AbstainPrecision","AbstainRecall","AbstainAccuracy_Softmax","AbstainAccuracy_Class","AbstainAccuracy_Random"], inplace=True)#"AbstainAccuracy_Variance","AbstainAccuracy_BALD"


    if method=="Vanilla":
        num_ensemble = 1


    df.to_csv(model_path + "_" + str(language)+"_"+str(layer_strategy) + "_"+str(last_layer_strategy)+"_"+ str(reduce_strategy) + "_"+str(decay_factor) +"_" + str(language_model_relation) + "_" + str(cased)+ "_" + str(bert_type) +"_" + str(learning_rate) +datetime.now().strftime("%Y%m%d-%H%M%S")+ "_random_state" + str(seed) +"_technique_" +str(method)+"_ensembles_"+str(num_ensemble)+".csv")


# 
# # Zero Shot

# In[ ]:


def abstain_accuracy_top_k(y_true,predictions, confidence, fraction, k, confidence_ordering="mean"):
    #the higher the confidence score, the higher the confidence
    if confidence_ordering=="mean":
        indices_top_k = np.argpartition(confidence, -k, axis=1)[:,::-1][:,:k]

        max_values_within = confidence[np.arange(confidence.shape[0])[:, None], indices_top_k]

        max_values_across = np.sum(max_values_within, axis=1)

        #get index of the max confidence for each testsample
        max_indices = np.argsort(max_values_across)

        #compute slicing fraction
        low_conf_indices_slice = int(fraction*len(max_indices))



        if (low_conf_indices_slice == 0 or low_conf_indices_slice == 1):
            indices = max_indices
        else:
            #select indices according to slice
            indices = max_indices[low_conf_indices_slice:]


    #the lower the confidence score, the higher the confidence --> variance
    elif confidence_ordering=="var":
        #print(confidence)
        #get the mean variance per sample, see how certain is your transformer
        mean_values = np.mean(confidence,axis=1)

        #print(mean_values)

        #get index of the max confidence for each testsample
        max_indices = np.argsort(mean_values)

        #print(max_indices)

        #compute slicing fraction
        low_conf_indices_slice = int(fraction*len(max_indices))

        #print(low_conf_indices_slice)


        if (low_conf_indices_slice == 0 or low_conf_indices_slice == 1):
            indices = max_indices
        else:
            #select indices according to slice
            indices = max_indices[:-low_conf_indices_slice]

    elif confidence_ordering=="BALD":
        #print("BALD")
        #get index of the max confidence for each testsample
        max_indices = np.argsort(confidence)

        #compute slicing fraction
        low_conf_indices_slice = int(fraction*len(max_indices))


        if (fraction == 0 or fraction == 1):
            indices = max_indices
        else:
            #select indices according to slice
            indices = max_indices[:-low_conf_indices_slice]

        #print(indices)

    elif confidence_ordering=="random":
        #get index of the max confidence for each testsample
        max_indices = np.argsort(confidence)


        #compute slicing fraction
        low_conf_indices_slice = int(fraction*len(max_indices))

        if (fraction == 0 or fraction == 1):
            indices = max_indices
        else:
            #select indices according to slice
            indices = max_indices[low_conf_indices_slice:]


    elif confidence_ordering=="class":

        confidence = get_class_ranking(confidence, k)

        #get index of the max confidence for each testsample
        max_indices = np.argsort(confidence)


        #compute slicing fraction
        low_conf_indices_slice = int(fraction*len(max_indices))

        if (fraction == 0 or fraction == 1):
            indices = max_indices
        else:
            #select indices according to slice
            indices = max_indices[low_conf_indices_slice:]

    #print("low_conf_indices_slice",low_conf_indices_slice)

    #print("indices",indices)


    #print("predictions",predictions)
    #predictions sliced
    predictions_sliced = np.take(predictions, indices, 0)

    #print("predictions_sliced sliced",predictions_sliced)

    y_true_sliced = y_true[indices]

    #print("y_true",y_true_sliced)

    acc = np.mean(tf.math.in_top_k(y_true_sliced,predictions_sliced, k=k))

    combined_accuracy = fraction*1+(1-fraction)*acc



    return acc, combined_accuracy


# In[ ]:


def get_class_ranking(confidence, k):
    indices_top_k = np.argpartition(confidence, -k, axis=1)[:,::-1][:,:k]

    unique, counts = np.unique(indices_top_k, return_counts=True)

    lookup_dict = dict(zip(unique,counts))

    class_count = [[lookup_dict[c] for c in s] for s in indices_top_k]
    class_count=np.array([np.array(xi) for xi in class_count])

    class_count_per_sample = np.sum(class_count, axis=1)


    support_ranking = scale(np.array(class_count_per_sample))

    return support_ranking


# In[ ]:


def compute_metrics_zero_shot(method,k,y_test,y_test_categorical,dropout_probs_samples,dropout_probs_mean, dropout_probs_var, num_ensemble, language, layer_strategy, last_layer_strategy,reduce_strategy,decay_factor,language_model_relation,cased,bert_type,learning_rate,seed, model_path):
        print("k:",k)
        d = defaultdict(dict)
        c = defaultdict(dict)
        b = defaultdict(dict)
        a = defaultdict(dict)
        e = defaultdict(dict)

        metrics_dict = {"ECE":None, "BRIER":None, "CalibAUROC":d, "CalibAUPR":c, "AbstainAccuracy":b, "AbstainPrecision":a,"AbstainRecall":e}

        #bald computation
        if method != "Vanilla":
            B = bald(dropout_probs_samples)

            bald_norm = scale(B)
            var_norm = scale(np.mean(dropout_probs_var,axis=1))

            bald_norm_inv = invert(bald_norm)
            var_norm_inv = invert(var_norm)

        #class ranking for class baseline
        #get index of the max confidence for each testsample
        max_indices = np.argmax(dropout_probs_mean,axis=1)

        unique, counts = np.unique(max_indices, return_counts=True)

        lookup_dict = dict(zip(unique,counts))
        
        support_ranking = get_class_ranking(dropout_probs_mean, k)


        random_ranking = np.random.rand(np.shape(y_test)[0])

        y_pred = np.argmax(dropout_probs_mean, axis=1) 
        
        indices_top_k = np.argpartition(dropout_probs_mean, -k, axis=1)[:,::-1][:,:k]


        max_values_within = dropout_probs_mean[np.arange(dropout_probs_mean.shape[0])[:, None], indices_top_k]

        max_values_across = np.sum(max_values_within, axis=1)


        confidence =  scale(max_values_across)
        ## ROC


        if method != "Vanilla":
            confidence_scores = [confidence, var_norm_inv, bald_norm_inv, support_ranking, random_ranking]
            for i,j in enumerate(["Softmax","Variance","BALD","Class","Random"]):
                metric = CalibrationAUC_Topk()
                metric.update_state(y_test,dropout_probs_mean,confidence_scores[i],k=k)
                metrics_dict["CalibAUROC"][j]=metric.result().numpy()

            for i,j in enumerate(["Softmax","Variance","BALD","Class","Random"]):
                metric = CalibrationAUC_Topk(curve="PR")
                metric.update_state(y_test,dropout_probs_mean,confidence_scores[i],k=k)
                metrics_dict["CalibAUPR"][j]=metric.result().numpy()


        elif method == "Vanilla":
            confidence_scores = [confidence, support_ranking, random_ranking]
            for i,j in enumerate(["Softmax","Class","Random"]):
                #print(j)
                metric = CalibrationAUC_Topk()
                metric.update_state(y_test,dropout_probs_mean,confidence_scores[i],k=k)
                metrics_dict["CalibAUROC"][j]=metric.result().numpy()

            for i,j in enumerate(["Softmax","Class","Random"]):
                metric = CalibrationAUC_Topk(curve="PR")
                metric.update_state(y_test,dropout_probs_mean,confidence_scores[i],k=k)
                metrics_dict["CalibAUPR"][j]=metric.result().numpy()

        fractions = np.arange(0, 1.001, 0.005)

        #dropout_probs_mean

        for i in fractions:
            metrics_dict["AbstainAccuracy"][i]["Softmax"] = abstain_accuracy_top_k(y_test,dropout_probs_mean,dropout_probs_mean, i,k=k ,confidence_ordering="mean")
            metrics_dict["AbstainAccuracy"][i]["Class"] = abstain_accuracy_top_k(y_test,dropout_probs_mean,dropout_probs_mean, i,k=k,confidence_ordering="class")
            metrics_dict["AbstainAccuracy"][i]["Random"] = abstain_accuracy_top_k(y_test,dropout_probs_mean,random_ranking, i,k=k,confidence_ordering="random")
            if method != "Vanilla":
                metrics_dict["AbstainAccuracy"][i]["Variance"] = abstain_accuracy_top_k(y_test,dropout_probs_mean,dropout_probs_var, i,k=k ,confidence_ordering="var")
                metrics_dict["AbstainAccuracy"][i]["BALD"] = abstain_accuracy_top_k(y_test,dropout_probs_mean,B, i,k=k,confidence_ordering="BALD")




            metric = AbstainPrecision_Topk(abstain_fraction= i)
            metric.update_state(y_test,dropout_probs_mean,confidence,k=k)
            metrics_dict["AbstainPrecision"][i]["Softmax"] = metric.result().numpy()

            metric = AbstainPrecision_Topk(abstain_fraction= i)
            metric.update_state(y_test,dropout_probs_mean,support_ranking,k=k)
            metrics_dict["AbstainPrecision"][i]["Class"] = metric.result().numpy()

            metric = AbstainPrecision_Topk(abstain_fraction= i)
            metric.update_state(y_test,dropout_probs_mean,random_ranking,k=k)
            metrics_dict["AbstainPrecision"][i]["Random"] = metric.result().numpy()

            if method != "Vanilla":
                metric = AbstainPrecision_Topk(abstain_fraction= i)
                metric.update_state(y_test,dropout_probs_mean,var_norm_inv,k=k)
                metrics_dict["AbstainPrecision"][i]["Variance"] = metric.result().numpy()

                metric = AbstainPrecision_Topk(abstain_fraction= i)
                metric.update_state(y_test,dropout_probs_mean,bald_norm_inv,k=k)
                metrics_dict["AbstainPrecision"][i]["BALD"] = metric.result().numpy()




            metric = AbstainRecall_Topk(abstain_fraction= i)
            metric.update_state(y_test,dropout_probs_mean,confidence,k=k)
            metrics_dict["AbstainRecall"][i]["Softmax"] = metric.result().numpy()

            metric = AbstainRecall_Topk(abstain_fraction= i)
            metric.update_state(y_test,dropout_probs_mean,support_ranking,k=k)
            metrics_dict["AbstainRecall"][i]["Class"] = metric.result().numpy()

            metric = AbstainRecall_Topk(abstain_fraction= i)
            metric.update_state(y_test,dropout_probs_mean,random_ranking,k=k)
            metrics_dict["AbstainRecall"][i]["Random"] = metric.result().numpy()
            if method != "Vanilla":
                metric = AbstainRecall_Topk(abstain_fraction= i)
                metric.update_state(y_test,dropout_probs_mean,var_norm_inv,k=k)
                metrics_dict["AbstainRecall"][i]["Variance"] = metric.result().numpy()

                metric = AbstainRecall_Topk(abstain_fraction= i)
                metric.update_state(y_test,dropout_probs_mean,bald_norm_inv,k=k)
                metrics_dict["AbstainRecall"][i]["BALD"] = metric.result().numpy()


        #metrics_dict

        df = pd.DataFrame.from_dict(metrics_dict)

        #df

        df["CalibAUROC Softmax"] = df["CalibAUROC"].iloc[0]
        if method != "Vanilla":
            df["CalibAUROC Variance"] = df["CalibAUROC"].iloc[1]
            df["CalibAUROC BALD"] = df["CalibAUROC"].iloc[2]
        df["CalibAUROC Class"] = df["CalibAUROC"].iloc[3]
        df["CalibAUROC Random"] = df["CalibAUROC"].iloc[4]

        df["CalibAUPR Softmax"] = df["CalibAUPR"].iloc[0]
        if method != "Vanilla":
            df["CalibAUPR Variance"] = df["CalibAUPR"].iloc[1]
            df["CalibAUPR BALD"] = df["CalibAUPR"].iloc[2]
        df["CalibAUPR Class"] = df["CalibAUPR"].iloc[3]
        df["CalibAUPR Random"] = df["CalibAUPR"].iloc[4]

        #df

        df.drop(columns=["CalibAUROC","CalibAUPR"],inplace=True)

        df.ffill(inplace=True)


        if method != "Vanilla":
            df = df.iloc[5:]
            df[["AbstainAccuracy_Softmax","AbstainAccuracy_Class","AbstainAccuracy_Random","AbstainAccuracy_Variance","AbstainAccuracy_BALD"]] = df["AbstainAccuracy"].apply(pd.Series)
            for i in ["AbstainAccuracy_Softmax","AbstainAccuracy_Class","AbstainAccuracy_Random","AbstainAccuracy_Variance","AbstainAccuracy_BALD"]:
                df[[i+"_Acc", i+"_CombinedAcc"]] = pd.DataFrame(df[i].tolist(), index=df.index)
            df[["AbstainPrecision_Softmax","AbstainPrecision_Class","AbstainPrecision_Random","AbstainPrecision_Variance","AbstainPrecision_BALD"]] = df["AbstainPrecision"].apply(pd.Series)
            df[["AbstainRecall_Softmax","AbstainRecall_Class","AbstainRecall_Random","AbstainRecall_Variance","AbstainRecall_BALD"]] = df["AbstainRecall"].apply(pd.Series)
            df.drop(columns=["AbstainAccuracy","AbstainPrecision","AbstainRecall","AbstainAccuracy_Softmax","AbstainAccuracy_Class","AbstainAccuracy_Random","AbstainAccuracy_Variance","AbstainAccuracy_BALD"], inplace=True)#"AbstainAccuracy_Variance","AbstainAccuracy_BALD"
        elif method == "Vanilla":
            df = df.iloc[3:]
            df[["AbstainAccuracy_Softmax","AbstainAccuracy_Class","AbstainAccuracy_Random"]] = df["AbstainAccuracy"].apply(pd.Series)
            for i in ["AbstainAccuracy_Softmax","AbstainAccuracy_Class","AbstainAccuracy_Random"]:
                df[[i+"_Acc", i+"_CombinedAcc"]] = pd.DataFrame(df[i].tolist(), index=df.index)
            df[["AbstainPrecision_Softmax","AbstainPrecision_Class","AbstainPrecision_Random"]] = df["AbstainPrecision"].apply(pd.Series)
            df[["AbstainRecall_Softmax","AbstainRecall_Class","AbstainRecall_Random"]] = df["AbstainRecall"].apply(pd.Series)
            df.drop(columns=["AbstainAccuracy","AbstainPrecision","AbstainRecall","AbstainAccuracy_Softmax","AbstainAccuracy_Class","AbstainAccuracy_Random"], inplace=True)#"AbstainAccuracy_Variance","AbstainAccuracy_BALD"


        if method=="Vanilla":
            num_ensemble = 1

        return df
        #df.to_csv("combined"+"_"  + str(k) +"_" + model_path + "_" + str(language)+"_"+str(layer_strategy) + "_"+str(last_layer_strategy)+"_"+ str(reduce_strategy) + "_"+str(decay_factor) +"_" + str(language_model_relation) + "_" + str(cased)+ "_" + str(bert_type) +"_" + str(learning_rate) +datetime.now().strftime("%Y%m%d-%H%M%S")+ "_random_state" + str(seed) +"_technique_" +str(method)+"_ensembles_"+str(num_ensemble)+".csv")

