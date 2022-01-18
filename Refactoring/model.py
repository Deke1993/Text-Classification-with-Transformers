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


def load_model(layer_strategy, bert_type, cased, language,language_model_relation):
    if layer_strategy != "last":
        output_hidden_states = True
        print('output hidden')
    else:
        output_hidden_states = False



    if bert_type == "BERT":
        if cased == False:
            # Load transformers config and set output_hidden_states to False
            if language != "All":

                language_size_model_dict = {"EN": {"specific":'bert-base-uncased', 
                                               "multi":'bert-base-multilingual-cased'}, 
                                            "DE":{"specific":"bert-base-german-cased", 
                                                  "multi":'bert-base-multilingual-cased'}, 
                                            "RO":{"multi":'bert-base-multilingual-cased'}, 
                                            "All":{"multi":'bert-base-multilingual-cased'}}

                config = BertConfig.from_pretrained(language_size_model_dict[language][language_model_relation])
                config.output_hidden_states = output_hidden_states
                tokenizer = BertTokenizer.from_pretrained(language_size_model_dict[language][language_model_relation], config = config)
                transformer_model = TFBertModel.from_pretrained(language_size_model_dict[language][language_model_relation], config = config)

            else:
                config = BertConfig.from_pretrained('bert-base-multilingual-cased')
                config.output_hidden_states = output_hidden_states
                tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', config=config)
                transformer_model = TFBertModel.from_pretrained('bert-base-multilingual-cased',config=config)

        else:
            if language != "All":

                language_size_model_dict = {"EN": {"specific":'bert-base-cased', 
                                               "multi":'bert-base-multilingual-cased'}, 
                                            "DE":{"specific":"bert-base-german-cased", 
                                                  "multi":'bert-base-multilingual-cased'}, 
                                            "RO":{"multi":'bert-base-multilingual-cased'}, 
                                            "All":{"multi":'bert-base-multilingual-cased'}}
                config = BertConfig.from_pretrained(language_size_model_dict[language][language_model_relation])
                config.output_hidden_states = output_hidden_states
                tokenizer = BertTokenizer.from_pretrained(language_size_model_dict[language][language_model_relation], config=config)
                transformer_model = TFBertModel.from_pretrained(language_size_model_dict[language][language_model_relation],config=config)

            else:
                config = BertConfig.from_pretrained('bert-base-multilingual-cased')
                config.output_hidden_states = output_hidden_states
                tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', config=config)
                transformer_model = TFBertModel.from_pretrained('bert-base-multilingual-cased',config=config)

    elif bert_type == "Albert":
        size_model_dict = {"small":"","medium":'albert-base-v2',"large":'albert-large-v2'}

        config = AlbertConfig.from_pretrained(size_model_dict[size])
        config.output_hidden_states = output_hidden_states
        tokenizer = AlbertTokenizer.from_pretrained(size_model_dict[size], config=config)
        transformer_model = TFAlbertModel.from_pretrained(size_model_dict[size],config=config)

    elif bert_type == "Roberta":
        size_model_dict = {"small":'roberta-base',"medium":'roberta-base',"large":'roberta-large'}

        config = RobertaConfig.from_pretrained(size_model_dict[size])
        config.output_hidden_states = output_hidden_states
        tokenizer = RobertaTokenizer.from_pretrained(size_model_dict[size],config=config)
        transformer_model = TFRobertaModel.from_pretrained(size_model_dict[size],config=config)

    elif bert_type == "RobertaXLM":
        config = XLMRobertaConfig.from_pretrained('jplu/tf-xlm-roberta-base')
        config.output_hidden_states = output_hidden_states
        tokenizer = XLMRobertaTokenizer.from_pretrained('jplu/tf-xlm-roberta-base', config=config)
        transformer_model = TFXLMRobertaModel.from_pretrained('jplu/tf-xlm-roberta-base', config=config)

    elif bert_type == "Distilbert":
        if cased == False:
            if language != "All":

                language_size_model_dict = {"EN": {"specific":'distilbert-base-uncased', 
                                               "multi":'distilbert-base-multilingual-cased'}, 
                                            "DE":{"specific":"distilbert-base-german-cased", 
                                                  "multi":'distilbert-base-multilingual-cased'}, 
                                            "RO":{"multi":'distilbert-base-multilingual-cased'}, 
                                            "All":{"multi":'distilbert-base-multilingual-cased'}}

                config = DistilBertConfig.from_pretrained(language_size_model_dict[language][language_model_relation])
                config.output_hidden_states = output_hidden_states
                tokenizer = BertTokenizer.from_pretrained(language_size_model_dict[language][language_model_relation], config=config)
                transformer_model = TFDistilBertModel.from_pretrained(language_size_model_dict[language][language_model_relation],config=config)

            else:
                config = DistilBertConfig.from_pretrained('distilbert-base-multilingual-cased')
                config.output_hidden_states = output_hidden_states
                tokenizer = BertTokenizer.from_pretrained('distilbert-base-multilingual-cased',config=config)
                transformer_model = TFDistilBertModel.from_pretrained('bert-base-multilingual-cased',config=config)

        else:
            if language != "All":

                language_size_model_dict = {"EN": {"specific":'distilbert-base-cased', 
                                               "multi":'distilbert-base-multilingual-cased'}, 
                                            "DE":{"specific":"distilbert-base-german-cased", 
                                                  "multi":'distilbert-base-multilingual-cased'}, 
                                            "RO":{"multi":'distilbert-base-multilingual-cased'}, 
                                            "All":{"multi":'distilbert-base-multilingual-cased'}}
                config = DistilBertConfig.from_pretrained(language_size_model_dict[language][language_model_relation])
                config.output_hidden_states = output_hidden_states
                tokenizer = BertTokenizer.from_pretrained(language_size_model_dict[language][language_model_relation], config=config)
                transformer_model = TFDistilBertModel.from_pretrained(language_size_model_dict[language][language_model_relation],config=config)

            else:
                config = DistilBertConfig.from_pretrained('distilbert-base-multilingual-cased')
                config.output_hidden_states = output_hidden_states
                tokenizer = BertTokenizer.from_pretrained('distilbert-base-multilingual-cased', config=config)
                transformer_model = TFDistilBertModel.from_pretrained('bert-base-multilingual-cased', config=config)

    print(tokenizer)
    print(transformer_model)
    
    return tokenizer, transformer_model, config


# In[ ]:


#implementation according to transformer library which implements the original setup as in paper
def build_classifier_model(seed, tokenizer, transformer_model, max_length, bert_type, reduce_strategy, config, n_classes):
    
    reduce = tf.math.reduce_mean if reduce_strategy=='mean' else tf.math.reduce_max
    
    transformer = transformer_model.layers[0]
    # Build your model input
    input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
    attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32') 
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    
    if bert_type=="Roberta":
        roberta_model = transformer(inputs)[0]#sequence output because pooler output not good for roberta
        if reduce_strategy == 'cls':
            pooled_output = roberta_model[:,0,:]
        else:
            pooled_output = reduce(roberta_model,axis=1)
        #basically what follows is what is done to compute pooled output
        dropout_pooler = Dropout(config.hidden_dropout_prob, name='pooler_dropout', seed=seed)
        pooled_output_dropout = dropout_pooler(pooled_output)
        hidden_roberta = Dense(units=config.hidden_size, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='hidden', activation='tanh')(pooled_output_dropout) #take s token
        #from here pooled output is used essentially
        dropout_hidden = Dropout(config.hidden_dropout_prob, name='hidden_output', seed=seed)
        hidden_output_roberta = dropout_hidden(hidden_roberta)
        output = Dense(units=n_classes, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='output')(hidden_output_roberta)
        
    elif bert_type=="Distilbert":
        distilbert_model = transformer(inputs)[0]
        if reduce_strategy == 'cls':
            pooled_output = distilbert_model[:,0]
        else:
            pooled_output = reduce(distilbert_model, axis=1)
        #for distilbert, the pooler is a bit different, e.g. relu activation
        hidden_layer = Dense(units=config.dim, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='hidden', activation='relu')(pooled_output)
        dropout_hidden = Dropout(config.seq_classif_dropout, name='hidden_output', seed=seed)
        hidden_output_distilbert = dropout_hidden(hidden_layer)
        output = Dense(units=n_classes, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='output')(hidden_output_distilbert)
        
    else:
        # Load the Transformers BERT model as a layer in a Keras model
        if reduce_strategy == 'cls':
            pooled_output = transformer(inputs)[1] #pooler output in tensorflow
        else:
            bert_model = transformer(inputs)[0] #pooler output in tensorflow
            pooled_output = reduce(bert_model,axis=1)
            dropout_pooler = Dropout(config.hidden_dropout_prob, name='pooler_dropout', seed=seed)
            pooled_output = dropout_pooler(pooled_output)
            pooled_output = Dense(units=config.hidden_size, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='hidden', activation='tanh')(pooled_output) #take s token
        # Then build your model output
        dropout = Dropout(config.hidden_dropout_prob, name='pooled_output', seed=seed)
        pooled_output = dropout(pooled_output) #training=False
        output = Dense(units=n_classes, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='output')(pooled_output)
            
    #outputs = {'output': output}
    # And combine it all in a model object
    model = Model(inputs=inputs, outputs=output, name="Text-Classifier")
    return model


# In[ ]:


#last4 concat strategy either cls, mean or max of respective hidden state
def build_classifier_model_last4(seed,tokenizer, transformer_model, max_length, bert_type, reduce_strategy, last_layer_strategy, config, n_classes):
    
    reduce = tf.math.reduce_mean if reduce_strategy=='mean' else tf.math.reduce_max
    reduce_cross_layer = tf.math.reduce_mean if last_layer_strategy=='mean' else tf.math.reduce_max
    
    transformer = transformer_model.layers[0]
    # Build your model input
    input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
    attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32') 
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    
    if bert_type=="Roberta":
        roberta_model = transformer(inputs)[2]#models hidden states
        if last_layer_strategy=="concat":
            pooled_output = tf.concat(tuple([roberta_model[i] for i in [-4, -3, -2, -1]]), axis=-1) #4 last hidden states
        else:
            pooled_output = reduce_cross_layer(tuple([roberta_model[i] for i in [-4, -3, -2, -1]]), axis=0) #4 last hidden states
        if reduce_strategy == 'cls':
            pooled_output = pooled_output[:, 0, :]
        else:
            pooled_output = reduce(pooled_output,axis=1)
        dropout_pooler = Dropout(config.hidden_dropout_prob, name='pooler_dropout', seed=seed)
        pooled_output = dropout_pooler(pooled_output)
        pooled_output = Dense(units=config.hidden_size, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='hidden', activation='tanh')(pooled_output) #take s token
        dropout_hidden = Dropout(config.hidden_dropout_prob, name='hidden_output', seed=seed)
        pooled_output = dropout_hidden(pooled_output)
        output = Dense(units=n_classes, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='output')(pooled_output)
        
    elif bert_type=="Distilbert":
        distilbert_model = transformer(inputs)[1]
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
        pooled_output = dropout_hidden(pooled_output)
        output = Dense(units=n_classes, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='output')(pooled_output)
        
    else:
    # Load the Transformers BERT model as a layer in a Keras model
        bert_hidden_states = transformer(inputs)[2] #hidden states
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
        pooled_output = dropout(pooled_output) #training=False
        pooled_output = Dense(units=config.hidden_size, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='hidden', activation='relu')(pooled_output)
        dropout_hidden = Dropout(config.hidden_dropout_prob, name='hidden_output', seed=seed)
        pooled_output = dropout_hidden(pooled_output)
        output = Dense(units=n_classes, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='output')(pooled_output)

    #outputs = {'output': output}
    # And combine it all in a model object
    model = Model(inputs=inputs, outputs=output, name="Text-Classifier")
    return model


# # Regular Optimizer

# In[4]:


#from huggingface
def bert_optimizer(learning_rate, 
                   batch_size, epochs,len_train, val_size=0.2, 
                   warmup_rate=0.1):
    """Creates an AdamWeightDecay optimizer with learning rate schedule."""
    train_data_size = len_train*(1-val_size)
  
    steps_per_epoch = int(train_data_size / batch_size)
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(warmup_rate * num_train_steps)  

    # Creates learning schedule.
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
      initial_learning_rate=learning_rate,
      decay_steps=num_train_steps,
      end_learning_rate=0.0)  
    
    warmup_schedule = optimization.WarmUp(
    initial_learning_rate=lr_schedule(num_warmup_steps),
    decay_schedule_fn = lr_schedule,
    warmup_steps=num_warmup_steps)
    
    return optimization.AdamWeightDecay( #warmup schedule ist not lr schedule
      learning_rate=warmup_schedule,
      weight_decay_rate=0.01, #0.01 is standard
      epsilon=1e-6,
      exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'])


# # Weight Decay

# In[2]:


def get_layers(model):
    if "DistilBert" in str(model.layers[2]):
        print("DistilBert")
        bert_embedding_layer_list = [s._layers for s in model.layers[2]._layers if "Embeddings" in str(s)][0]
        bert_main_layer_list = [s._layers[0] for s in model.layers[2]._layers if "Transformer" in str(s)][0][:][::-1]
        classifier_layers = list(model.layers[3:])
        return [classifier_layers,*bert_main_layer_list,bert_embedding_layer_list]
    else:
        print("BERT")
        bert_embedding_layer_list = [s._layers for s in model.layers[2]._layers if "Embeddings" in str(s)][0]
        bert_main_layer_list = [s._layers[0] for s in model.layers[2]._layers if "Encoder" in str(s)][0][::-1]
        #bert_pooler_layer_list = [s for s in model.layers[2]._layers if "Pooler" in str(s)][0]
        classifier_layers = list(model.layers[3:])
        return [classifier_layers,*bert_main_layer_list,bert_embedding_layer_list]
    
    #[classifier_layers,bert_pooler_layer_list,*bert_main_layer_list,bert_embedding_layer_list]


# In[3]:


def dlr_optimizer(learning_rate, layer_list, decay_factor, batch_size, epochs, len_train, val_size):
    decay_list = []
    for i,x in enumerate(layer_list):
        if i == 0:
            decay_list.append(learning_rate)
        else:
            decay_list.append(decay_list[i-1]*decay_factor)
            
    optimizers = [bert_optimizer(decay_list[i], batch_size, epochs, len_train, val_size) for i in range(len(decay_list))]
    
    optimizers_and_layers_main_bert = list(zip(optimizers, layer_list))
    
    return optimizers_and_layers_main_bert


# In[ ]:




