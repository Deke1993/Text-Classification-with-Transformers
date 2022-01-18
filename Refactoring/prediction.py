#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from Refactoring.model import load_model, build_classifier_model, build_classifier_model_last4
import pandas as pd
import numpy as np
import tensorflow as tf


# In[ ]:


def make_prediction(k,free_text:list, model_path:str, tokenizer, model_params:list,method=None, threshold=None) -> list:
    max_length = len(max(free_text, key=len))
    
    seed, tokenizer, transformer_model, bert_type, reduce_strategy, layer_strategy ,last_layer_strategy, config, n_classes, label_dict = model_params
    
    test_x = tokenizer(
    text=free_text,
    add_special_tokens=True,
    max_length=max_length,
    padding='max_length', 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)


    # Take a look at the model
    if layer_strategy == 'last':
        model = build_classifier_model(seed=seed, tokenizer=tokenizer, transformer_model=transformer_model, max_length=max_length,bert_type=bert_type, reduce_strategy=reduce_strategy, config=config, n_classes=n_classes)
    elif layer_strategy == 'last4':
        model = build_classifier_model_last4(seed=seed, tokenizer=tokenizer, transformer_model=transformer_model, max_length=max_length,bert_type=bert_type, reduce_strategy=reduce_strategy, last_layer_strategy=last_layer_strategy, config=config, n_classes=n_classes)
    model.summary()
    
    # The model weights (that are considered the best) are loaded into the model.
    model.load_weights(model_path)
    
    predictions_logit = model.predict(x={'input_ids': test_x['input_ids'],'attention_mask':test_x["attention_mask"]})
    
    y_pred = np.argmax(predictions_logit, axis=1) 
    
    print(tf.nn.softmax(predictions_logit))
    
    inv_label_dict = {v: k for k, v in label_dict.items()}
    
    codes = [inv_label_dict[y_pred[i]] for i in range(len(y_pred))]
    
    map_df = pd.read_excel("material_groups_map.XLSX", header=0)
    
    map_df = map_df.set_index("material group")
    
    top_pred = map_df.loc[codes,"Material Group Desc."]
    
    topk = [predictions_logit[i].argsort()[-k:][::-1] for i in range(len(y_pred))]
    
    topk_list = [[inv_label_dict[l] for l in i] for i in topk]
    
    topk_return = [map_df.loc[i,"Material Group Desc."].to_list() for i in topk_list]
    
    return top_pred, topk_return

