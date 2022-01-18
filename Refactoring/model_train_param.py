#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load Huggingface transformers
from transformers import TFBertModel,  BertConfig, BertTokenizer
from transformers import TFAlbertModel,  AlbertConfig, AlbertTokenizer
from transformers import TFRobertaModel,  RobertaConfig, RobertaTokenizer
from transformers import TFDistilBertModel, BertTokenizer, DistilBertConfig
from transformers import TFXLMModel, XLMTokenizer, XLMConfig, TFSequenceSummary
from transformers import TFXLMRobertaModel, XLMRobertaTokenizer, XLMRobertaConfig

# tensorflow.keras
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
import tensorflow as tf
from official.nlp import optimization  # to create AdamW optimizer
import official.nlp.modeling.layers as layers
import tensorflow_addons as tfa


# In[ ]:


#implementation according to transformer library which implements the original setup as in paper
def build_classifier_model_training(training, seed, tokenizer, transformer_model, max_length, bert_type, reduce_strategy, config, n_classes):
    
    reduce = tf.math.reduce_mean if reduce_strategy=='mean' else tf.math.reduce_max
    
    transformer = transformer_model.layers[0]
    # Build your model input
    input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
    attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32') 
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    
    if bert_type=="Roberta":
        roberta_model = transformer(inputs, training=training)[0]#sequence output because pooler output not good for roberta
        if reduce_strategy == 'cls':
            pooled_output = roberta_model[:,0,:]
        else:
            pooled_output = reduce(roberta_model,axis=1)
        #basically what follows is what is done to compute pooled output
        dropout_pooler = Dropout(config.hidden_dropout_prob, name='pooler_dropout', seed=seed)
        pooled_output_dropout = dropout_pooler(pooled_output, training=training)
        hidden_roberta = Dense(units=config.hidden_size, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='hidden', activation='tanh')(pooled_output_dropout) #take s token
        #from here pooled output is used essentially
        dropout_hidden = Dropout(config.hidden_dropout_prob, name='hidden_output', seed=seed)
        hidden_output_roberta = dropout_hidden(hidden_roberta, training=training)
        output = Dense(units=n_classes, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='output')(hidden_output_roberta)
        
    elif bert_type=="Distilbert":
        distilbert_model = transformer(inputs, training=training)[0]
        if reduce_strategy == 'cls':
            pooled_output = distilbert_model[:,0]
        else:
            pooled_output = reduce(distilbert_model, axis=1)
        #for distilbert, the pooler is a bit different, e.g. relu activation
        hidden_layer = Dense(units=config.dim, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='hidden', activation='relu')(pooled_output)
        dropout_hidden = Dropout(config.seq_classif_dropout, name='hidden_output', seed=seed)
        hidden_output_distilbert = dropout_hidden(hidden_layer, training=training)
        output = Dense(units=n_classes, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='output')(hidden_output_distilbert)
        
    else:
        # Load the Transformers BERT model as a layer in a Keras model
        if reduce_strategy == 'cls':
            pooled_output = transformer(inputs, training=training)[1] #pooler output in tensorflow
        else:
            bert_model = transformer(inputs, training=training)[0] #pooler output in tensorflow
            pooled_output = reduce(bert_model,axis=1)
            dropout_pooler = Dropout(config.hidden_dropout_prob, name='pooler_dropout', seed=seed)
            pooled_output = dropout_pooler(pooled_output, training=training)
            pooled_output = Dense(units=config.hidden_size, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='hidden', activation='tanh')(pooled_output) #take s token
        # Then build your model output
        dropout = Dropout(config.hidden_dropout_prob, name='pooled_output', seed=seed)
        pooled_output = dropout(pooled_output, training=training) #training=False
        output = Dense(units=n_classes, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='output')(pooled_output)
            
    #outputs = {'output': output}
    # And combine it all in a model object
    model = Model(inputs=inputs, outputs=output, name="Text-Classifier")
    return model


# In[ ]:


#last4 concat strategy either cls, mean or max of respective hidden state
def build_classifier_model_last4_training(training, seed,tokenizer, transformer_model, max_length, bert_type, reduce_strategy, last_layer_strategy, config, n_classes):
    
    reduce = tf.math.reduce_mean if reduce_strategy=='mean' else tf.math.reduce_max
    reduce_cross_layer = tf.math.reduce_mean if last_layer_strategy=='mean' else tf.math.reduce_max
    
    transformer = transformer_model.layers[0]
    # Build your model input
    input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
    attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32') 
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    
    if bert_type=="Roberta":
        roberta_model = transformer(inputs, training=training)[2]#models hidden states
        if last_layer_strategy=="concat":
            pooled_output = tf.concat(tuple([roberta_model[i] for i in [-4, -3, -2, -1]]), axis=-1) #4 last hidden states
        else:
            pooled_output = reduce_cross_layer(tuple([roberta_model[i] for i in [-4, -3, -2, -1]]), axis=0) #4 last hidden states
        if reduce_strategy == 'cls':
            pooled_output = pooled_output[:, 0, :]
        else:
            pooled_output = reduce(pooled_output,axis=1)
        dropout_pooler = Dropout(config.hidden_dropout_prob, name='pooler_dropout', seed=seed)
        pooled_output = dropout_pooler(pooled_output, training=training)
        pooled_output = Dense(units=config.hidden_size, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='hidden', activation='tanh')(pooled_output) #take s token
        dropout_hidden = Dropout(config.hidden_dropout_prob, name='hidden_output', seed=seed)
        pooled_output = dropout_hidden(pooled_output, training=training)
        output = Dense(units=n_classes, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='output')(pooled_output)
        
    elif bert_type=="Distilbert":
        distilbert_model = transformer(inputs, training=training)[1]
        if last_layer_strategy=="concat":
            pooled_output = tf.concat(tuple([distilbert_model[i] for i in [-2, -1]]), axis=-1)
        else:
            pooled_output = reduce_cross_layer(tuple([distilbert_model[i] for i in [-2, -1]]), axis=0)
        if reduce_strategy == 'cls':
            pooled_output = pooled_output[:, 0, :]
        else:
            pooled_output = reduce(pooled_output,axis=1)
        #no dropout here because pooler is not using dropout in distilbert
        #dropout_pooler = Dropout(config.seq_classif_dropout, name='pooler_dropout', seed=seed)
        #pooled_output = dropout_pooler(pooled_output)
        pooled_output = Dense(units=config.dim, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='hidden', activation='relu')(pooled_output)
        dropout_hidden = Dropout(config.seq_classif_dropout, name='hidden_output', seed=seed)
        pooled_output = dropout_hidden(pooled_output, training=training)
        output = Dense(units=n_classes, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='output')(pooled_output)
        
    else:
    # Load the Transformers BERT model as a layer in a Keras model
        bert_hidden_states = transformer(inputs, training=training)[2] #hidden states
        # Then build your model output
        if last_layer_strategy=="concat":
            pooled_output = tf.concat(tuple([bert_hidden_states[i] for i in [-4, -3, -2, -1]]), axis=-1)
        else:
            pooled_output = reduce_cross_layer(tuple([bert_hidden_states[i] for i in [-4, -3, -2, -1]]), axis=0)
        if reduce_strategy == 'cls':
            pooled_output = pooled_output[:, 0, :]
        else:
            pooled_output = reduce(pooled_output,axis=1)
        dropout = Dropout(config.hidden_dropout_prob, name='pooled_output', seed=seed)
        pooled_output = dropout(pooled_output, training=training) #training=False
        pooled_output = Dense(units=config.hidden_size, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='hidden', activation='relu')(pooled_output)
        dropout_hidden = Dropout(config.hidden_dropout_prob, name='hidden_output', seed=seed)
        pooled_output = dropout_hidden(pooled_output, training=training)
        output = Dense(units=n_classes, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='output')(pooled_output)

    #outputs = {'output': output}
    # And combine it all in a model object
    model = Model(inputs=inputs, outputs=output, name="Text-Classifier")
    return model


# In[ ]:




