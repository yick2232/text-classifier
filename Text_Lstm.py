import sys
sys.path.append('/home/justin/sklearn')
from tensowflowing import LoadKmsDataUtils
from tensowflowing import model_config
from tensowflowing import utils
from tensowflowing import NormalEmbedding
import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Embedding, Bidirectional, LSTM
from tensorflow.keras import layers as L
from collections import Counter
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import Model
from gensim.models import Word2Vec


class HaltCondition(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(logs)
        if(logs.get('accuracy') > 0.995):
            self.model.stop_training =  True
        if(logs.get('accuracy') > 0.1):
            self.inference_by_lstm()
        print('模型准确率为{}'.format(logs.get('accuracy')))

    def inference_by_lstm(self):
        test_data_path = '../sklearning/test_data/测试集模板_v5.csv'
        sententsTransformer = utils.load_model(model_config.text_lstm_x)
        label_encoder = utils.load_model(model_config.lr_y_path)
        df = pd.read_csv(test_data_path)

        x, y = LoadKmsDataUtils.get_x_y(df)
        x = sententsTransformer.encoder(x)
        y = label_encoder.transform(y)

        x = tf.constant(x)
        y = tf.constant(y)

        loss, accuracy =  self.model.evaluate(x, y, verbose=2)
        print('模型损失值{}, 模型精确率={},样本数量={}'.format(loss, accuracy, len(x)))

def build_lstm(sequence_length, num_classes, vocab_size, embedding_size,embedding_matrix=None):
    input_x = L.Input(shape=(sequence_length,), name='input_x')

    # embedding layer
    if embedding_matrix is None:
        embedding = L.Embedding(vocab_size, embedding_size, name='embedding')(input_x)
    else:
        embedding = L.Embedding(vocab_size, embedding_size, weights=[embedding_matrix], name='embedding')(input_x)

    lstm = LSTM(num_classes * 4, activation='sigmoid')
    bilstm = Bidirectional(lstm)
    output_bi = bilstm(embedding)
    dropout = tf.keras.layers.Dropout(0.3)(output_bi)
    o1 = tf.keras.layers.Dense(num_classes * 2, activation='relu')(dropout)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(o1)

    model = Model(input_x, output)
    model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

class SententsTransformer():
    def __init__(self, sequence_length = None):
        self.word2idx = None
        self.w2v_embeddings = None
        self.sequence_length = sequence_length

    def build_vocab_dict(self, sentences, vocab_size=2000):
        counter = Counter()
        word2idx = dict()
        for sent in sentences:
            for char in sent:
                counter[char] += 1
        word2idx = dict()
        word2idx['<unk>'] = 0
        if vocab_size > 0:
            num_most_common = vocab_size - len(word2idx)
        else:
            num_most_common = len(counter)
        for char, _ in counter.most_common(num_most_common):
            word2idx[char] = word2idx.get(char, len(word2idx))
        self.word2idx = word2idx
        return word2idx

    def word2vector(self, x, vocab_size, embedding_size):
        '''整个文档word2vec 训练'''
        w2v_train_sents = []
        for doc in x:
            w2v_train_sents.append(list(doc))
        w2v_model = Word2Vec(w2v_train_sents, vector_size=embedding_size)
        print(w2v_model.wv.key_to_index)
        '''将训练数据向量编码'''
        w2v_embeddings = np.zeros((vocab_size, embedding_size))
        for char, char_idx in self.word2idx.items():
            if char in w2v_model.wv:
                w2v_embeddings[char_idx] = w2v_model.wv[char]

        self.w2v_embeddings = w2v_embeddings
        return w2v_embeddings

    def encoder(self, x):
        train_x = []
        for doc in x:
            item = []
            for char in doc:
                if self.word2idx.get(char) == None:
                    item.append(self.word2idx['<unk>'])
                else:
                    item.append(self.word2idx[char])

            for i in range(0, self.sequence_length - len(item)):
                item.append(self.word2idx['<unk>'])
            train_x.append(item)
        x = tf.constant(train_x)
        return x

if __name__=='__main__':
    x, y, y_encoder, _ = LoadKmsDataUtils.get_train_data(useTfidf=False)
    vocab_size = 15000
    sequence_length = 80
    embedding_size = 100

    sententsTransformer = SententsTransformer(sequence_length)
    sententsTransformer.build_vocab_dict(x, vocab_size)
    w2v_embeddings = sententsTransformer.word2vector(x, vocab_size, embedding_size)
    x = sententsTransformer.encoder(x)
    classes_num = len(y_encoder.classes_)
    utils.save_model(model_config.lr_y_path, y_encoder)
    utils.save_model(model_config.text_lstm_x, sententsTransformer)

    '''
    sequence_length = 200
    embedding_size = 100
    embedding = NormalEmbedding.NormalEmbedding(fill_len=sequence_length)
    x = embedding.fit_transform(x)
    
    utils.save_model(model_config.lr_y_path, y_encoder)
    utils.save_model(model_config.text_lstm_x, embedding)
    
    x = tf.constant(x)
    vocab_size = embedding.get_word_num()
    
    classes_num = len(y_encoder.classes_)
    '''

    lstm_model = build_lstm(sequence_length, classes_num, vocab_size, embedding_size, w2v_embeddings)
    print(lstm_model.summary())

    haltCondition = HaltCondition()
    lstm_model.fit(x, y, batch_size=64, epochs=80, callbacks=[haltCondition])

    loss, accuracy = lstm_model.evaluate(x, y, verbose=2)
    print('模型损失值{}, 模型精确率={}'.format(loss, accuracy))
    lstm_model.save(model_config.text_lstm_model)

    start_time = datetime.now()
    a = lstm_model.predict(x)
    p_y = np.argmax(a, axis=1)
    c = [1 if p_y[i] == v else 0 for i, v in enumerate(y)]
    c_label = y_encoder.inverse_transform(p_y)
    error_text = []
    error_label = []
    x_, y_, _, _ = LoadKmsDataUtils.get_train_data(useTfidf=False)
    for i,v in enumerate(c):
        if v == 0:
            error_text.append(x_[i])
            error_label.append(c_label[i])

    print('accuracy={}, 耗时{}'.format(sum(c)/len(c), datetime.now()-start_time))

