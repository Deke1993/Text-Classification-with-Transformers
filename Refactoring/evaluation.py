#!/usr/bin/env python
# coding: utf-8

# In[4]:


import csv
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from sklearn.metrics import classification_report
import os
import numpy as np
import pandas as pd

#loggging
from datetime import datetime


# In[2]:


def test_prediction(model,tokenizer, X_test, y_test, n_classes, max_length):
    #######################################
    ### ----- Evaluate the model ------ ###
    # Ready test data
    test_y_material = to_categorical(y_test,num_classes=n_classes)
    test_x = tokenizer(
        text=list(X_test),
        add_special_tokens=True,
        max_length=max_length,
        truncation=False,
        padding="max_length", 
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = True,
        verbose = True)
    predictions_logit = model.predict(x={'input_ids': test_x['input_ids'],'attention_mask':test_x["attention_mask"]})
    y_pred = np.argmax(predictions_logit, axis=1)
    
    return y_pred, predictions_logit, test_y_material


# In[3]:


def save_report(y_test,y_pred,n_classes,label_dict,inv_label_dict, language, layer_strategy, last_layer_strategy, reduce_strategy, decay_factor, language_model_relation, cased, bert_type, learning_rate, epochs, seed, test_lang):
    report_dict = classification_report(y_test, y_pred, labels=list(inv_label_dict.keys()), target_names=list(label_dict.keys()), output_dict=True)
    df_report = pd.DataFrame(report_dict)
    if language != "All":
        df_report.to_csv('report_' + str(language) +"_"+str(layer_strategy)+ "_"+str(last_layer_strategy) +"_"+ str(reduce_strategy) + "_"+str(decay_factor) + "_" + str(language_model_relation) + "_" + str(cased)+ "_" + str(bert_type) +"_" + str(learning_rate) +"_"+ str(epochs) +"_" +datetime.now().strftime("%Y%m%d-%H%M%S")+ "_random_state" + str(seed) + ".csv", index = True)
    else:
        df_report.to_csv('test on' + str(test_lang) + "_"+ 'report_' + str(language) +"_"+str(layer_strategy)+ "_"+str(last_layer_strategy) +"_"+ str(reduce_strategy) + "_"+str(decay_factor) + "_" + str(language_model_relation) + "_" + str(cased)+ "_" + str(bert_type) +"_" + str(learning_rate) +"_"+ str(epochs) +"_" +datetime.now().strftime("%Y%m%d-%H%M%S")+ "_random_state" + str(seed) + ".csv", index = True)

    
    
def save_metrics(test_y_material, predictions_logit,n_classes,label_dict,inv_label_dict, language, layer_strategy, last_layer_strategy, reduce_strategy, decay_factor, language_model_relation, cased, bert_type, learning_rate, epochs, seed, test_lang):
    metrics_dict = {"accuracy":None, "top2accuracy":None, "top3accuracy":None, "top4accuracy":None, "top5accuracy":None, "ROC":None, "PR":None}

    metrics = ["accuracy", "top2accuracy", "top3accuracy", "top4accuracy", "top5accuracy"]
    for i in [1,2,3,4,5]:  
        metric = tf.keras.metrics.TopKCategoricalAccuracy(k=i)
        metric.update_state(test_y_material,predictions_logit)
        metric.result().numpy()
        metrics_dict[metrics[i-1]]=metric.result().numpy()


    curves = ["ROC", "PR"]
    predictions = np.array(tf.nn.softmax(predictions_logit))
    for curve in curves:
        m = tf.keras.metrics.AUC(num_thresholds=50, curve=curve)
        m.update_state(test_y_material,predictions)
        value = m.result().numpy()
        metrics_dict[curve]=value


    if language != 'All':
        with open(str(language)+"_"+str(layer_strategy) + "_"+str(last_layer_strategy)+"_"+ str(reduce_strategy) + "_"+str(decay_factor) +"_" + str(language_model_relation) + "_" + str(cased)+ "_" + str(bert_type) +"_" + str(learning_rate) +datetime.now().strftime("%Y%m%d-%H%M%S")+ "_random_state" + str(seed)+'_dict.csv', 'w') as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in metrics_dict.items():
                writer.writerow([key, value])
    else:
        with open('test on' + str(test_lang) + "_" +str(language)+"_"+str(layer_strategy) + "_"+str(last_layer_strategy)+"_"+ str(reduce_strategy) + "_"+str(decay_factor) +"_" + str(language_model_relation) + "_" + str(cased)+ "_" + str(bert_type) +"_" + str(learning_rate) +datetime.now().strftime("%Y%m%d-%H%M%S")+ "_random_state" + str(seed)+'_dict.csv', 'w') as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in metrics_dict.items():
                writer.writerow([key, value])

    print(metrics_dict)


# In[ ]:




