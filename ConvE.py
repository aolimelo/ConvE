from linkpredictor import LinkPredictor
from keras.layers import Dense, Dropout, Embedding, Activation, Merge, Input, merge, Flatten, Lambda, \
    Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Conv3D, MaxPooling3D, Reshape, LocallyConnected1D, LocallyConnected2D, \
    AveragePooling2D
from keras.models import Sequential, Model
from linkpredictor import LinkPredictor
import math
from keras.optimizers import Adam, Adagrad, RMSprop
import numpy as np
from collections import defaultdict
import random
from sample import type_index, LCWASampler, CorruptedSampler, RandomModeSampler, SOSampler
from scipy.interpolate import interp2d
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback
import threading
from keras.constraints import maxnorm, nonneg, unitnorm
from keras.regularizers import l1, l2
from tools import ConvEEval, ranking_scores, huang2003_layers, RankCallback
from copy import deepcopy, copy
from keras_layers import SimpleCombinationOperator, MatrixCombinationOperator,  CircularCorrelation, CircularCorrelation2D, max_margin


class ConvE(LinkPredictor):
    def __init__(self, n_dim, n_relations, n_instances, filter_dim=3, stride=1, max_pool_dim=2, activation="relu",
                 dropout=0.25, n_neg=15, sample_mode="so", epochs=100, constraint=0, regularizer=0.0001,
                 rank_callback=False,
                 batch_size=200, combination="simple", initialization=None, n_channels=1, verbose=0, layers=None,
                 patience=10):
        # self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.sess = tf.Session()
        K.set_session(self.sess)
        self.activation = activation
        self.dropout = dropout
        self.n_neg = n_neg
        self.sample_mode = sample_mode
        self.training_epochs = epochs
        self.batch_size = batch_size
        self.combination = combination
        self.initialization = initialization
        self.verbose = verbose
        self.layers = layers
        self.filter_dim = filter_dim
        self.stride = stride
        self.n_channels = n_channels
        self.max_pool_dim = max_pool_dim
        self.emb_side_dim = int(math.sqrt(n_dim))
        self.dimensions = self.emb_side_dim ** 2
        self.emb_shape = (self.emb_side_dim, self.emb_side_dim, self.n_channels)
        self.n_relations = n_relations
        self.n_instances = n_instances
        self.patience = patience
        self.lock = threading.Lock()
        self.rank_callback = rank_callback
        self.constraint = maxnorm(constraint, axis=1) if constraint else None
        self.regularizer = l1(regularizer) if regularizer else None
        self.border_mode = "valid"
        self.callbacks = []
        self.callbacks.append(TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False))
        self.callbacks.append(
            EarlyStopping(monitor='val_loss', patience=self.patience, verbose=self.verbose, mode='auto'))
        self.callbacks.append(
            ModelCheckpoint(filepath="/tmp/%s.hdf5" % self.__class__.__name__, verbose=1, save_best_only=True))

    def combine_so(self, s, o):
        if self.combination == "simple":
            x = SimpleCombinationOperator()([s, o])
        elif self.combination == "matrix":
            s = Reshape(self.emb_shape[:-1], input_shape=self.emb_shape)(s)
            o = Reshape(self.emb_shape[:-1], input_shape=self.emb_shape)(o)
            x = MatrixCombinationOperator()([s, o])
            x = Reshape(self.emb_shape, input_shape=self.emb_shape[:-1])(x)
        elif self.combination == "ccorr":
            x = CircularCorrelation()([Flatten()(s), Flatten()(o)])
            x = Reshape(self.emb_shape,input_shape=(self.dimensions*self.n_channels,))(x)
        elif self.combination == "ccorr2d":
            s = Reshape(self.emb_shape[:-1], input_shape=self.emb_shape)(s)
            o = Reshape(self.emb_shape[:-1], input_shape=self.emb_shape)(o)
            x = CircularCorrelation2D()([s,o])
            x = Reshape(self.emb_shape, input_shape=self.emb_shape[:-1])(x)
        elif self.combination == "concat":
            x = merge([s, o], mode="concat", name=self.combination)
        elif self.combination == "diff":
            x = Lambda(lambda x: x[0] - x[1], name=self.combination)([s, o])
        elif self.combination == "mult":
            x = Lambda(lambda x: x[0] * x[1], name=self.combination)([s, o])
        elif self.combination == "dot":
            x = merge([s, o], mode="dot", name=self.combination, dot_axes=(1, 2))
        else:
            raise ("Combination operator %s is not supported" % self.combination)
        return x

    def create_sampler(self, triples, sz):
        if self.sample_mode == "so":
            sampler = SOSampler(self.n_neg, triples, sz)
        elif self.sample_mode == 'corrupted':
            ti = type_index(triples)
            sampler = CorruptedSampler(self.n_neg, triples, sz, ti)
        elif self.sample_mode == 'random':
            sampler = RandomModeSampler(self.n_neg, [0, 1], triples, sz)
        elif self.sample_mode == 'lcwa':
            sampler = LCWASampler(self.n_neg, [0, 1], triples, sz)
        return sampler

    def add_model_layer(self, x, size):
        x = Conv2D(size, self.filter_dim, self.filter_dim, subsample=(self.stride, self.stride),
                   border_mode = self.border_mode, activation=self.activation, W_constraint=self.constraint)(x)
        x = MaxPooling2D((self.max_pool_dim, self.max_pool_dim), border_mode=self.border_mode)(x)
        x = Activation("relu")(x)
        return x

    def create_model(self, train_triples=None, valid_triples=None):
        if self.layers is None:
            self.layers = huang2003_layers(self.dimensions, self.n_relations)[-1:]

        print("%d classes, %d instances, %d dim embeddings with hidden layers = %s" % (
            self.n_relations, self.n_instances, self.dimensions, str(self.layers)))

        self.input_s = input_s = Input(shape=(1,), dtype="int32", name="input_s")
        self.input_o = input_o = Input(shape=(1,), dtype="int32", name="input_o")

        embeddings = Embedding(input_dim=self.n_instances, output_dim=self.dimensions * self.n_channels, trainable=True,
                               W_constraint=self.constraint)

        subject_emb = embeddings(input_s)
        object_emb = embeddings(input_o)

        subject_emb = Reshape(self.emb_shape, input_shape=(self.dimensions * self.n_channels,))(
            subject_emb)
        object_emb = Reshape(self.emb_shape, input_shape=(self.dimensions * self.n_channels,))(
            object_emb)

        x = self.combine_so(subject_emb, object_emb)

        for i, hidden_units in enumerate(self.layers):
            x = self.add_model_layer(x, hidden_units)

        x = Flatten()(x)
        x = Dense(self.n_relations, W_regularizer=self.regularizer, W_constraint=self.constraint)(x)
        predictions = Activation("sigmoid")(x)

        self.model = Model(input=[self.input_s, self.input_o], output=predictions)
        self.model.compile(loss='binary_crossentropy', optimizer=Adam())
        self.model.summary()

        _, self.rank = tf.nn.top_k(tf.transpose(predictions), k=self.n_instances)

    def convert_data(self, triples, sampler=None):
        data = defaultdict(lambda: np.zeros((self.n_relations,)))
        for s, o, p in triples:
            data[(s, o)][p] = 1
        if self.n_neg and sampler is not None:
            r_negs = sampler.sample(zip(triples, [1] * len(triples)))
            for (s, o, p), y in r_negs:
                data[(s, o)] = data[(s, o)]

        x_s = np.array([s for s, o in data.keys()], dtype="int32").reshape((-1, 1))
        x_o = np.array([o for s, o in data.keys()], dtype="int32").reshape((-1, 1))
        Y = np.array([y for y in data.values()])

        return x_s, x_o, Y

    def batch_generator(self, train_triples, sampler):
        while 1:
            x_s, x_o, Y = self.convert_data(train_triples, sampler)
            idx = np.arange(Y.shape[0])
            np.random.shuffle(idx)
            for i in range(idx.shape[0] // self.batch_size):
                batch_idx = idx[i * self.batch_size:(i + 1) * self.batch_size]
                yield [x_s[batch_idx], x_o[batch_idx]], Y[batch_idx]

    def fit_generator(self, train_triples, valid_triples):
        random.seed(88), np.random.seed(88)
        self.create_model(train_triples, valid_triples)

        sz = (self.n_instances, self.n_instances, self.n_relations)
        all_triples = train_triples + valid_triples

        sampler = self.create_sampler(all_triples, sz)
        v_sampler = self.create_sampler(all_triples, sz)

        validation_x_s, validation_x_o, validation_y = self.convert_data(valid_triples, sampler=v_sampler)

        if self.rank_callback:
            self.callbacks.append(
                RankCallback(ConvEEval(valid_triples, all_triples, verbose=self.verbose), self, patience=1))

        self.model.fit_generator(self.batch_generator(train_triples, sampler),
                                 samples_per_epoch=len(train_triples) * (self.n_neg + 1),
                                 nb_epoch=self.training_epochs, nb_worker=1, max_q_size=10, verbose=self.verbose,
                                 callbacks=self.callbacks,
                                 validation_data=([validation_x_s, validation_x_o], validation_y))

    def fit(self, train_triples, valid_triples):

        random.seed(88), np.random.seed(88)
        self.create_model(train_triples, valid_triples)

        sz = (self.n_instances, self.n_instances, self.n_relations)
        all_triples = train_triples + valid_triples

        sampler = self.create_sampler(all_triples, sz)
        v_sampler = self.create_sampler(all_triples, sz)

        training_x_s, training_x_o, training_y = self.convert_data(train_triples, sampler=sampler)
        validation_x_s, validation_x_o, validation_y = self.convert_data(valid_triples, sampler=sampler)

        if self.rank_callback:
            self.callbacks.append(
                RankCallback(ConvEEval(valid_triples, all_triples, verbose=self.verbose), self, patience=1))

        self.model.fit([training_x_s, training_x_o], training_y,
                       batch_size=self.batch_size, nb_epoch=self.training_epochs,
                       verbose=self.verbose, shuffle=True, callbacks=self.callbacks,
                       validation_data=([validation_x_s, validation_x_o], validation_y))

    def predict_proba(self, triples):
        y_idx = []
        ss = []
        oo = []
        for i, (s, o, p) in enumerate(triples):
            y_idx.append(p)
            ss.append(s)
            oo.append(o)
        ss = np.array(ss, dtype="int32").reshape((-1, 1))
        oo = np.array(oo, dtype="int32").reshape((-1, 1))

        scores = self.model.predict([ss, oo])
        x_idx = range(len(triples))
        return scores[x_idx, y_idx]

    def predict(self, triples):
        return self.predict_proba(triples) > 0.5

    def save_model(self, path):
        self.model.save(path + ".h5")