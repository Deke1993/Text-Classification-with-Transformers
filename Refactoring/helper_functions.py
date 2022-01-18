#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


#when dealing with artificial ood data, the trained model has fewer output classes (204) 
#than necessary when being evaluated on the test data. Also the class numbering is not the same.
#Therefore, this function maps the reduced class numbers to the correct one (rearrange the columns of predictiosn)
#and inserts columns with all zeros for classes the model has never seen during training (those that were delted)
def map_logits(label_dict_train, label_dict_zero, predictions_logits):
    index_dict = {}
    count_in = 0
    count_out = 0
    count = 0
    for k, v in label_dict_train.items():
        if k in label_dict_zero:
            count_in += 1
            index_dict[count] = label_dict_zero[k]
        else:

            count_out += 1
        count += 1

    
    values = index_dict.values()
    
    keys = list(index_dict.keys())
    
    predictions_logit_en = np.zeros((np.shape(predictions_logits)[0], len(label_dict_zero)))
    
    columns = predictions_logits.take(keys,axis=1)
    
    predictions_logit_en[:,list(values)] = columns
    
    return predictions_logit_en


# In[ ]:


#functions excpect between 0 and 1, and the higher the more confidenct: https://github.com/google/uncertainty-baselines/blob/main/baselines/toxic_comments/metrics.py
def scale(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# In[ ]:


def invert(data):
    return (-data) + 1

