#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import random


# In[8]:


def load(language, seed):
    df = pd.read_csv("/media/remote/jdeke/Dataframes/df_combined.csv", header=0)
    
    #filter using language and only keep the two columns of interest
    df = df[df["language key"]==language][["short text","material group"]]

    #make sure text is string
    df["short text"] = df["short text"].astype(str)

    df = df.groupby('material group').filter(lambda x : len(x) > 1) # filter out classes occuring only once

    #get unique material groups
    possible_labels = df["material group"].unique()

    #create a mapping from class to integer 
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index

    #get number of unique classes
    n_classes = len(label_dict)

    #replace class name with integer represnting class
    df['label'] = df["material group"].replace(label_dict)

    #do train test split, train-val split will be done in .fit
    X_train, X_test, y_train, y_test = train_test_split(df["short text"].values, 
                                                      df.label.values, 
                                                      test_size=0.2, 
                                                        random_state=seed,
                                                        stratify=df["label"], shuffle=True
                                                      )
    
    
    
    return X_train, X_test, y_train, y_test, n_classes, label_dict


# In[9]:


#if language == "All"
def load_combined_languages(seed):
    df = pd.read_csv("/media/remote/jdeke/Dataframes/df_combined.csv", header=0)
    
    #do not filter by language
    df = df[["short text","material group", "language key"]]

    df["short text"] = df["short text"].astype(str)

    df = df.groupby('material group').filter(lambda x : len(x) > 1) # filter out classes occuring only once

    possible_labels = df["material group"].unique()

    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index

    n_classes = len(label_dict)

    df['label'] = df["material group"].replace(label_dict)
    
    #split up in language dfs, necessary to create test-sets to evaluate multilingual models on each language later on
    df_DE = df[df["language key"]=="DE"][["short text","material group","label"]]
    df_EN = df[df["language key"]=="EN"][["short text","material group","label"]]
    df_RO = df[df["language key"]=="RO"][["short text","material group","label"]]
    
    df_DE = df_DE.groupby('material group').filter(lambda x : len(x) > 1) # filter out classes occuring only once
    df_EN = df_EN.groupby('material group').filter(lambda x : len(x) > 1) # filter out classes occuring only once
    df_RO = df_RO.groupby('material group').filter(lambda x : len(x) > 1) # filter out classes occuring only once
    
    X_train_de, X_test_de, y_train_de, y_test_de = train_test_split(df_DE["short text"].values, 
                                                  df_DE.label.values, 
                                                  test_size=0.2, 
                                                    random_state=seed,
                                                    stratify=df_DE["label"], shuffle=True
                                                  )

    X_train_en, X_test_en, y_train_en, y_test_en = train_test_split(df_EN["short text"].values, 
                                                      df_EN.label.values, 
                                                      test_size=0.2, 
                                                        random_state=seed,
                                                        stratify=df_EN["label"], shuffle=True
                                                      )

    X_train_ro, X_test_ro, y_train_ro, y_test_ro = train_test_split(df_RO["short text"].values, 
                                                      df_RO.label.values, 
                                                      test_size=0.2, 
                                                        random_state=seed,
                                                        stratify=df_RO["label"], shuffle=True
                                                      )
    
    n_classes_en = len(np.unique(y_test_en))
    n_classes_de = len(np.unique(y_test_de))
    n_classes_ro = len(np.unique(y_test_ro))
    
    #create combined train and test sets
    X_train = np.concatenate((X_train_de,X_train_en,X_train_ro))
    y_train = np.concatenate((y_train_de,y_train_en,y_train_ro))
    
    #shuffle to not have first one language, then the second and then the last as a strict order
    X_train, y_train = shuffle(X_train, y_train, random_state=seed)
    
    return X_train, y_train, X_train_de, X_test_de, y_train_de, y_test_de, X_train_en, X_test_en, y_train_en, y_test_en, X_train_ro, X_test_ro, y_train_ro, y_test_ro, n_classes, label_dict


# In[ ]:


def load_artificial_ood(language, seed):
    df = pd.read_csv("/media/remote/jdeke/Dataframes/df_combined.csv", header=0)
    
    #only applicable with specific language
    df = df[df["language key"]==language][["short text","material group"]]
    
    df["short text"] = df["short text"].astype(str)
    
    df = df.groupby('material group').filter(lambda x : len(x) > 1) # filter out classes occuring only once
    
    classes_A = df["material group"].unique()
    

    if language=="EN":
        #get unique classes of german
        df_other = pd.read_csv("/media/remote/jdeke/Dataframes/df_combined.csv", header=0)
        df_other = df_other[df_other["language key"]=="DE"][["short text","material group"]]
        df_other["short text"] = df_other["short text"].astype(str)

        df_other = df_other.groupby('material group').filter(lambda x : len(x) > 1) # filter out classes occuring only once

        classes_B = df_other["material group"].unique()
        
    elif language=="DE":
        #get unique classes of english
        df_other = pd.read_csv("/media/remote/jdeke/Dataframes/df_combined.csv", header=0)
        df_other = df_other[df_other["language key"]=="EN"][["short text","material group"]]
        df_other["short text"] = df_other["short text"].astype(str)

        df_other = df_other.groupby('material group').filter(lambda x : len(x) > 1) # filter out classes occuring only once

        classes_B = df_other["material group"].unique()

    #list of items in A that are not in B
    classes_delete = list(set(classes_A) - set(classes_B)) 
    #print(len(classes_delete))
    
    # delete those items from df
    df = df.loc[~df['material group'].isin(classes_delete)]
    
    possible_labels = df["material group"].unique()

    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    
    df['label'] = df["material group"].replace(label_dict)
    
    n_classes = len(label_dict)
    
    X_train, X_test, y_train, y_test = train_test_split(df["short text"].values, 
                                                  df.label.values, 
                                                  test_size=0.2, 
                                                    random_state=seed,
                                                    stratify=df["label"], shuffle=True
                                                  )
    
    return X_train, X_test, y_train, y_test, n_classes, label_dict


# In[ ]:


def get_reduced_label_dict(language):
    df = pd.read_csv("/media/remote/jdeke/Dataframes/df_combined.csv", header=0)
    df = df[df["language key"]==language][["short text","material group"]]


    df["short text"] = df["short text"].astype(str)

    df = df.groupby('material group').filter(lambda x : len(x) > 1) # filter out classes occuring only once

    possible_labels = df["material group"].unique()

    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    label_dict

    n_classes = len(label_dict)

    df['label'] = df["material group"].replace(label_dict)
    
    classes_A = df["material group"].unique()

    if language=="EN":
        #get unique classes of german
        df_other = pd.read_csv("/media/remote/jdeke/Dataframes/df_combined.csv", header=0)
        df_other = df_other[df_other["language key"]=="DE"][["short text","material group"]]
        df_other["short text"] = df_other["short text"].astype(str)

        df_other = df_other.groupby('material group').filter(lambda x : len(x) > 1) # filter out classes occuring only once

        classes_B = df_other["material group"].unique()
        
    elif language=="DE":
        #get unique classes of english
        df_other = pd.read_csv("/media/remote/jdeke/Dataframes/df_combined.csv", header=0)
        df_other = df_other[df_other["language key"]=="EN"][["short text","material group"]]
        df_other["short text"] = df_other["short text"].astype(str)

        df_other = df_other.groupby('material group').filter(lambda x : len(x) > 1) # filter out classes occuring only once

        classes_B = df_other["material group"].unique()
        
    classes_delete = list(set(classes_A) - set(classes_B)) 

    df = df.loc[~df['material group'].isin(classes_delete)]
    
    n_classes_reduced = len(df["label"].unique())
    
    possible_labels_reduced = df["material group"].unique()

    label_dict_reduced = {}
    for index, possible_label in enumerate(possible_labels_reduced):
        label_dict_reduced[possible_label] = index
    
    return label_dict_reduced, n_classes_reduced


# In[ ]:


def get_train_dict(language):
    df = pd.read_csv("/media/remote/jdeke/Dataframes/df_combined.csv", header=0)
    df = df[df["language key"]==language][["short text","material group"]]

    df["short text"] = df["short text"].astype(str)

    df = df.groupby('material group').filter(lambda x : len(x) > 1) # filter out classes occuring only once

    possible_labels = df["material group"].unique()

    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    
    n_classes = len(label_dict)
        
    return label_dict, n_classes

