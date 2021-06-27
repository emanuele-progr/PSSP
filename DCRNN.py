
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from keras.models import Model, Input
from keras.models import load_model
from keras.layers import Embedding, Dense, TimeDistributed, Concatenate, BatchNormalization, Masking, LSTM
from keras.layers import Bidirectional, Activation, Dropout, GRU, Conv1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import seaborn as sn
from sklearn.model_selection import train_test_split, KFold
from keras.metrics import categorical_accuracy
from keras import backend as K
from sklearn.metrics import confusion_matrix
from keras.regularizers import l1, l2
import tensorflow as tf
import gzip

cb513 = np.load('cb513+profile_split1.npy.gz')
print(cb513.shape)
cb6133filtered = np.load('cullpdb+profile_6133_filtered.npy.gz')
columns = ["id", "len", "input", "profiles", "expected"]
maxlen_seq = r = 700  # protein residues padded to 700
f = 57  # number of features for each residue

def print_data(fit):

    #  "Accuracy"
    fig = plt.figure()
    plt.plot(fit.history['Q8_accuracy'])
    plt.plot(fit.history['val_Q8_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #plt.show()
    fig.savefig('accuracy.png')
    # "Loss"
    fig = plt.figure()
    plt.plot(fit.history['loss'])
    plt.plot(fit.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #plt.show()
    fig.savefig('loss.png')

def get_data(arr, bounds=None):
    if bounds is None: bounds = range(len(arr))

    data = [None for i in bounds]

    for i in bounds:
        seq, q8, profiles = '', '', []
        for j in range(r):
            jf = j * f

            # Residue convert from one-hot to decoded
            residue_onehot = arr[i, jf + 0:jf + 22]
            residue = residue_list[np.argmax(residue_onehot)]

            # Q8 one-hot encoded to decoded structure symbol
            residue_q8_onehot = arr[i, jf + 22:jf + 31]
            residue_q8 = q8_list[np.argmax(residue_q8_onehot)]

            if residue == 'NoSeq': break  # terminating sequence symbol

            nc_terminals = arr[i, jf + 31:jf + 33]  # nc_terminals = [0. 0.]
            sa = arr[i, jf + 33:jf + 35]  # sa = [0. 0.]
            profile = arr[i, jf + 35:jf + 57]  # profile features

            seq += residue  # concat residues into amino acid sequence
            q8 += residue_q8  # concat secondary structure into secondary structure sequence
            profiles.append(profile)


        data[i] = [str(i + 1), len(seq), seq, np.array(profiles), q8]

    return pd.DataFrame(data, columns=columns)


def show_secondary(array):

    to_img = np.copy(array)
    for i in range(to_img.shape[0]):
        for j in range(to_img.shape[1]):
            for k in range(to_img.shape[2]):
                if to_img[i, j, k] == 1:
                    to_img[i, j, k] = 255
    #img = Image.fromarray(to_img)
    plt.figure(figsize=(28, 10))
    plt.imshow(np.transpose(to_img[7, :, :]))
    plt.show()

def make_confusion_matrix(yt, yp):
    fig = plt.figure()
    #matrix = confusion_matrix(yt, yp, normalize='true')
    matrix = confusion_matrix(yt, yp)
    df_cm = pd.DataFrame(matrix, ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T'], ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T'])
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.0)  # for label size
    #sn.heatmap(df_cm, annot=True, fmt ='.2f', annot_kws={"size": 10})  # font size
    sn.heatmap(df_cm, annot=True, fmt='g', annot_kws={"size": 10})  # font size
    recall = np.diag(matrix)/np.sum(matrix, axis=1)
    precision = np.diag(matrix)/np.sum(matrix, axis=0)

    print(recall)
    print(precision)

    plt.show()
    fig.savefig('confusion_matrix.png')

    return matrix

# Convert probabilities or one_hot to secondary structure string sequences
def to_seq(y):
    seqs=[]
    for i in range(len(y)):
        seq_i=''
        for j in range(len(y[i])):
            if np.sum(y[i, j, :]) != 0:
                seq_i += q8_list[np.argmax(y[i][j])]
        seqs.append(seq_i)
    return seqs

def to_int_seq(y1, y2):
    seqs=[]
    for i in range(len(y1)):

        for j in range(len(y1[i])):
            if np.sum(y1[i, j, :]) != 0:
                seq_i = np.argmax(y2[i][j])
                seqs.append(seq_i)

    return seqs

def probabilities_to_onehot(y):

    one_hot = np.zeros(y.shape)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if np.sum(y[i, j, :]) > 0:
                one_hot[i, j, np.argmax(y[i, j, :])] = 1
    return one_hot

def categorical_crossentropy(y_true, y_pred):

    return K.mean(K.categorical_crossentropy(y_pred, y_true))

def conv_block(x, activation=True, batch_norm=True, drop_out=True, res=True):
    cnn = Conv1D(64, 11, padding="same")(x)
    if activation: cnn = TimeDistributed(Activation("relu"))(cnn)
    if batch_norm: cnn = TimeDistributed(BatchNormalization())(cnn)
    if drop_out:   cnn = TimeDistributed(Dropout(0.5))(cnn)
    if res:        cnn = Concatenate(axis=-1)([x, cnn])

    return cnn


def super_conv_block(x):
    c3 = Conv1D(32, 1, padding="same")(x)
    c3 = TimeDistributed(Activation("relu"))(c3)
    c3 = TimeDistributed(BatchNormalization())(c3)

    c7 = Conv1D(64, 3, padding="same")(x)
    c7 = TimeDistributed(Activation("relu"))(c7)
    c7 = TimeDistributed(BatchNormalization())(c7)

    c11 = Conv1D(128, 5, padding="same")(x)
    c11 = TimeDistributed(Activation("relu"))(c11)
    c11 = TimeDistributed(BatchNormalization())(c11)

    x = Concatenate(axis=-1)([x, c3, c7, c11])
    x = TimeDistributed(Dropout(0.5))(x)
    return x

# The custom accuracy metric used for this task

def Q8_accuracy(y_true, y_pred):
    y = tf.argmax(y_true, axis=- 1)
    y_ = tf.argmax(y_pred, axis=- 1)
    mask = tf.greater(y, 0)
    print (K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx()))
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())

def accuracy_Q8(real, pred):
    total = real.shape[0] * real.shape[1]
    correct = 0
    for i in range(real.shape[0]):
        for j in range(real.shape[1]):
            if np.sum(real[i, j, :]) == 0:
                total = total - 1
            else:
                if real[i, j, np.argmax(pred[i, j, :])] > 0:
                    correct = correct + 1

    print(total)
    return correct / total
def count_noSeq(X):
    total = X.shape[0] * X.shape[1]
    count = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j, 30] == 1:
                count = count + 1
    return count

def accuracy_Q8variant(real, pred):
    total = real.shape[0] * real.shape[1]
    correct = 0
    for i in range(real.shape[0]):
        for j in range(real.shape[1]):
            if np.sum(real[i, j, :]) == 0:
                if np.sum(pred[i, j, :]) == 0:
                    correct = correct + 1
            else:
                if real[i, j, np.argmax(pred[i, j, :])] > 0:
                    correct = correct + 1

    print(total)
    return correct / total
################################################################
""" train data processing """
################################################################

dataset_raw_input = np.load('cullpdb+profile_6133_filtered.npy.gz')
dataset = np.reshape(dataset_raw_input, (5534, 700, 57))
del dataset_raw_input

features_data = dataset[:, :, 0:21]

profile_data = dataset[:, :, 35:56]
s_structure_label = dataset[:, :, 22:30]

s_accessibility_label = dataset[:, :, 33:35]

residue_list = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L',
                'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X', 'NoSeq']
q8_list = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T', 'NoSeq']

aminoacids_lenght = 700 # protein residues padded to 700
features = 57 # number of features for each residue
feature_index = np.hstack((np.arange(0, 21), np.arange(35, 56)))  # 42-d features
X = dataset[:, :, feature_index]
#print(count_noSeq(dataset))
print(X.shape)
print(X[200][699])
vals = np.arange(0, 8)
num_seqs, seqlen, feature_dim = np.shape(X)

num_classes = len(q8_list) - 1

""" build secondary structure label"""
labels_new = np.zeros((num_seqs, seqlen))
for i in range(np.size(s_structure_label, axis=0)):
    labels_new[i, :] = np.dot(s_structure_label[i, :, :], vals)
labels_new = labels_new.astype('int32')
s_structure_label2 = labels_new


""" build solvent accessibility label"""
labels_new_2 = np.zeros((num_seqs, seqlen))
vals2 = np.array([2, 1])
for i in range(np.size(s_accessibility_label, axis=0)):
    labels_new_2[i, :] = np.dot(s_accessibility_label[i, :, :], vals2)
labels_new_2 = labels_new_2.astype('int32')
s_accessibility_label = labels_new_2
print(s_accessibility_label[200])
""" shuffle data"""
seq_shuffle = np.arange(0, num_seqs)
np.random.shuffle(seq_shuffle)
X_train = X[seq_shuffle[0:5022]]
s_struct_train = s_structure_label[seq_shuffle[0:5022]]
s_acc_train = s_accessibility_label[seq_shuffle[0:5022]]
features_data = features_data[seq_shuffle[0:5022]]
profile_data = profile_data[seq_shuffle[0:5022]]


print(profile_data[12][12])


################################################################
"""validation data processing"""
################################################################

X_valid = X[seq_shuffle[5022:5534]]
s_struct_valid = s_structure_label[seq_shuffle[5022:5534]]
s_acc_valid = s_accessibility_label[seq_shuffle[5022:5534]]

################################################################
"""test data processing"""
################################################################

testset_raw_input = np.load('cb513+profile_split1.npy.gz')
testset = np.reshape(testset_raw_input, (514, 700, 57))

del testset_raw_input

s_structure_label = testset[:, :, 22:30]

s_accessibility_label = testset[:, :, 33:35]
feature_index = np.hstack((np.arange(0, 21), np.arange(35, 56)))  # 42-d features
X = testset[:, :, feature_index]

# getting meta
num_seqs, seqlen, feature_dim = np.shape(X)
num_classes = 8

vals = np.arange(0, 8)

""" build secondary structure label"""
labels_new = np.zeros((num_seqs, seqlen))
for i in range(np.size(s_structure_label, axis=0)):
    labels_new[i, :] = np.dot(s_structure_label[i, :, :], vals)
labels_new = labels_new.astype('int32')
s_structure_label2 = labels_new

""" build solvent accessibility label"""
labels_new_2 = np.zeros((num_seqs,seqlen))
vals2 = np.array([2, 1])
for i in range(np.size(s_accessibility_label, axis=0)):
    labels_new_2[i, :] = np.dot(s_accessibility_label[i, :, :], vals2)
labels_new_2 = labels_new_2.astype('int32')
s_accessibility_label = labels_new_2

X_test = X
s_struct_test = s_structure_label
s_acc_test = s_accessibility_label

################################################################
"""model"""
################################################################

input_1 = Input(shape=(700, 21))
input_2 = Input(shape=(700, 21))

#input_1_reshape = Reshape(-1, 21)(input_1)
#input_2_reshape = Reshape(-1, 21)(input_2)

#input1_masked = Masking(mask_value=0., input_shape=(700, 21))(input_1)
#input2_masked = Masking(mask_value=0., input_shape=(700, 21))(input_2)


input_1_embedded = Dense(50, activation='relu')(input_1)

input_concat = Concatenate(axis=-1)([input_1_embedded, input_2])

x = super_conv_block(input_concat)
x = conv_block(x)
x = super_conv_block(x)
x = conv_block(x)
x = super_conv_block(x)
x = conv_block(x)

#input = Reshape(( 700, 21+50, 1))(input_concat)

#input = Reshape(( 700, 21+50))(input_concat)

lstm1 = Bidirectional(GRU(128,
                         return_sequences='True',
                         activation='tanh',
                         recurrent_activation='sigmoid',
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         recurrent_initializer='orthogonal',
                         bias_initializer='zeros',
                         dropout=0.3,
                         recurrent_dropout=0,
                         implementation=1))(x)



lstm2 = Bidirectional(GRU(128,
                         return_sequences='True',
                         activation='tanh',
                         recurrent_activation='sigmoid',
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         recurrent_initializer='orthogonal',
                         bias_initializer='zeros',
                         dropout=0.3,
                         recurrent_dropout=0,
                         implementation=1))(lstm1)

lstm3 = Bidirectional(GRU(128,
                         return_sequences='True',
                         activation='tanh',
                         recurrent_activation='sigmoid',
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         recurrent_initializer='orthogonal',
                         bias_initializer='zeros',
                         dropout=0.3,
                         recurrent_dropout=0,
                         implementation=1))(lstm2)

local_and_global = Concatenate(axis=-1)([lstm3, x])
x = TimeDistributed(Dense(256,
                          activation='relu',
                          use_bias=True,
                          kernel_initializer='glorot_uniform',
                          bias_initializer='zeros'))(local_and_global)
x = Dropout(0.5)(x)

x = TimeDistributed(Dense(128,
                          activation='relu',
                          use_bias=True,
                          kernel_initializer='glorot_uniform',
                          bias_initializer='zeros'))(x)
x = Dropout(0.5)(x)

x = TimeDistributed(Dense(128,
                          activation='relu',
                          use_bias=True,
                          kernel_initializer='glorot_uniform',
                          bias_initializer='zeros'))(x)

x = Dropout(0.5)(x)
#print(lstm3)
out_1 = TimeDistributed(Dense(8, activation="softmax"))(x)
#out_1 = Reshape((700, 8))(out_1)
'''

"""3 convolution layer with different kernel size"""

# kernel size 3
conv1 = ZeroPadding2D((3//2, 0), data_format='channels_last')(input)
conv1 = Conv2D(filters=64,
               kernel_size=(3, 21+50),
               input_shape=(1, 700, 21+50),
               data_format='channels_last',
               strides= (1,1),
               #dilation_rate=(1, 1),
               padding='valid',
               activation='relu',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros')(conv1)
#conv1 = tf.transpose(conv1, (0, 2, 3, 1))
#conv1 = Reshape((700, 64))(conv1)
conv1 = BatchNormalization(axis=-1)(conv1)
print(conv1)
# kernel size 7
conv2 = ZeroPadding2D((7//2, 0), data_format='channels_last')(input)

conv2 = Conv2D(filters=64,
               kernel_size=(7, 21+50),
               input_shape=(1, 700, 21+50),
               data_format='channels_last',
               strides= (1,1),
               padding='valid',
               #dilation_rate=(1, 1),
               activation='relu',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros')(conv2)
#conv2 = tf.transpose(conv2, (0, 2, 3, 1))
#conv2 = Reshape((700, 64))(conv2)
conv2 = BatchNormalization(axis=-1)(conv2)


# kernel size 11
conv3 = ZeroPadding2D((11//2, 0), data_format='channels_last')(input)
conv3 = Conv2D(filters=64,
               kernel_size=(11, 21+50),
               input_shape=(1, 700, 21+50),
               data_format='channels_last',
               strides=(1,1),
               padding='valid',
               #dilation_rate=1,
               activation='relu',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros')(conv3)
#conv3 = tf.transpose(conv3, (0, 2, 3, 1))
#conv3 = Reshape((700, 64))(conv3)
conv3 = BatchNormalization(axis=-1)(conv3)

# concat 3 conv

conv = concatenate([conv1, conv2, conv3], axis=3)
conv = BatchNormalization(axis=-1)(conv)
conv = Reshape((700, 64*3))(conv)
#mask_layer=Masking(mask_value=0., input_shape=(700, 191))(conv)


# Defining 3 bidirectional GRU layers; taking the concatenation of outputs
gru1 = Bidirectional(GRU(300,
                         return_sequences='True',
                         go_backwards='True',
                         activation='tanh',
                         recurrent_activation='sigmoid',
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         recurrent_initializer='orthogonal',
                         bias_initializer='zeros',
                         dropout=0.5,
                         recurrent_dropout=0))(conv)



gru2 = Bidirectional(GRU(300,
                         return_sequences='True',
                         go_backwards='True',
                         activation='tanh',
                         recurrent_activation='sigmoid',
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         recurrent_initializer='orthogonal',
                         bias_initializer='zeros',
                         dropout=0.5,
                         recurrent_dropout=0))(gru1)

gru3 = Bidirectional(GRU(300,
                         return_sequences='True',
                         go_backwards='True',
                         activation='tanh',
                         recurrent_activation='sigmoid',
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         recurrent_initializer='orthogonal',
                         bias_initializer='zeros',
                         dropout=0.5,
                         recurrent_dropout=0))(gru2)

conv_and_gru = concatenate([gru3, conv], axis=2)
# Defining two fully-connected layers with dropout
x = TimeDistributed(Dense(256,
                          activation='relu',
                          use_bias=True,
                          kernel_initializer='glorot_uniform',
                          bias_initializer='zeros'))(conv_and_gru)
x = Dropout(0.5)(x)

x = TimeDistributed(Dense(128,
                          activation='relu',
                          use_bias=True,
                          kernel_initializer='glorot_uniform',
                          bias_initializer='zeros'))(x)
x = Dropout(0.5)(x)
print(conv_and_gru)
out_1 = TimeDistributed(Dense(8, activation="softmax"))(x)
out_1 = Reshape((700, 8))(out_1)
print(out_1)
'''
model = Model((input_1, input_2), out_1)
model.summary()
for i, l in enumerate(model.layers):
    print(f'layer {i}: {l}')
    print(f'has input mask: {l.input_mask}')
    print(f'has output mask: {l.output_mask}')
#mask_layer=Masking(mask_value=0., input_shape=(timesteps, features))(inputs)
#print(mask_layer._keras_mask)

model.compile(optimizer="nadam", loss="categorical_crossentropy", metrics=Q8_accuracy)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose= 1)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
saved_model = load_model('best_model.h5', custom_objects={"Q8_accuracy" : Q8_accuracy})
#pred = saved_model.predict((X_test[:, :, 0:21], X_test[:, :, 21:42]))
#one_hot_predictions = probabilities_to_onehot(pred)
#tf.compat.v1.disable_eager_execution()
#acc = Q8_accuracy(s_struct_test, pred)
#print ('accuracy on cb513:', tf.compat.v1.Session().run(acc).mean())
#print(accuracy_Q8(s_struct_test, pred))
fit = model.fit((X_train[:, :, 0:21], X_train[:, :, 21:42]), s_struct_train,
                validation_data=((X_valid[:, :, 0:21], X_valid[:, :, 21:42]), s_struct_valid), callbacks=[es, mc], batch_size=64, epochs=300, verbose=1)
y_pre = model.predict((X_test[:, :, 0:21], X_test[:, :, 21:42]))

one_hot_predictions = probabilities_to_onehot(y_pre)
#print(Q8_accuracy(s_struct_test, one_hot_predictions).mean())

yt = to_int_seq(s_struct_test, s_struct_test)
yp = to_int_seq(s_struct_test, one_hot_predictions)
yt = list(map(str, yt))
yp = list(map(str, yp))
print_data(fit)
print(make_confusion_matrix(yt, yp))
#show_secondary(s_struct_test)
#plt.savefig('true_secondary.png')
#show_secondary(y_pre)
#plt.savefig('predicted_secondary.png')
#print_data(fit)
#print(K.get_value(Q8_accuracy(s_struct_test, y_pre)))

'''

#### alternate data preprocessing ####

# Maps the sequence to a one-hot encoding
def onehot_to_seq(oh_seq, index):
    s = ''
    for o in oh_seq:
        i = np.argmax(o)
        if i != 0:
            s += index[i]
        else:
            break
    return s


def seq2onehot(seq, n):
    out = np.zeros((len(seq), maxlen_seq, n))
    for i in range(len(seq)):
        for j in range(maxlen_seq):
            out[i, j, seq[i, j]] = 1
    return out


# Computes and returns the n-grams of a particualr sequence, defaults to trigrams
def seq2ngrams(seqs, n=1):
    return np.array([[seq[i: i + n] for i in range(len(seq))] for seq in seqs])


valid_input_seqs, valid_target_seqs = \
    valid_df[['input', 'expected']][(valid_df.len.astype(int) <= maxlen_seq)].values.T
valid_input_grams = seq2ngrams(valid_input_seqs)


# Loading and converting the inputs to trigrams
train_input_seqs, train_target_seqs = \
    train_df[['input', 'expected']][(train_df.len.astype(int) <= maxlen_seq)].values.T

train_input_grams = seq2ngrams(train_input_seqs)


# Same for test
test_input_seqs, test_target_seqs = \
    test_df[['input', 'expected']][(test_df.len.astype(int) <= maxlen_seq)].values.T

test_input_grams = seq2ngrams(test_input_seqs)

# Initializing and defining the tokenizer encoders and decoders based on the train set
tokenizer_encoder = Tokenizer()
tokenizer_encoder.fit_on_texts(train_input_grams)
tokenizer_decoder = Tokenizer(char_level=True)
tokenizer_decoder.fit_on_texts(train_target_seqs)

# Using the tokenizer to encode and decode the sequences for use in training
# Inputs

train_input_data = tokenizer_encoder.texts_to_sequences(train_input_grams)

train_input_data = sequence.pad_sequences(train_input_data,
                                          maxlen=maxlen_seq, padding='post')
valid_input_data = tokenizer_encoder.texts_to_sequences(valid_input_grams)
valid_input_data = sequence.pad_sequences(valid_input_data,
                                          maxlen=maxlen_seq, padding='post')

valid_target_data = tokenizer_decoder.texts_to_sequences(valid_target_seqs)
valid_target_data = sequence.pad_sequences(valid_target_data,
                                           maxlen=maxlen_seq, padding='post')
valid_target_data = to_categorical(valid_target_data)
# Targets
train_target_data = tokenizer_decoder.texts_to_sequences(train_target_seqs)
train_target_data = sequence.pad_sequences(train_target_data,
                                           maxlen=maxlen_seq, padding='post')
train_target_data = to_categorical(train_target_data)

test_target_data = tokenizer_decoder.texts_to_sequences(test_target_seqs)
test_target_data = sequence.pad_sequences(test_target_data,
                                           maxlen=maxlen_seq, padding='post')
test_target_data = to_categorical(test_target_data)
# Use the same tokenizer defined on train for tokenization of test
test_input_data = tokenizer_encoder.texts_to_sequences(test_input_grams)
test_input_data = sequence.pad_sequences(test_input_data,
                                         maxlen=maxlen_seq, padding='post')

# Computing the number of words and number of tags for the keras model
n_words = len(tokenizer_encoder.word_index) + 1
n_tags = len(tokenizer_decoder.word_index) + 1

valid_input_data_alt = valid_input_data
valid_input_data = seq2onehot(valid_input_data, n_words)
valid_profiles = valid_df.profiles.values

train_input_data_alt = train_input_data
train_input_data = seq2onehot(train_input_data, n_words)
train_profiles = train_df.profiles.values

test_input_data_alt = test_input_data
test_input_data = seq2onehot(test_input_data, n_words)
test_profiles = test_df.profiles.values

train_profiles_np = np.zeros((len(train_profiles), maxlen_seq, 22))
for i, profile in enumerate(train_profiles):
    for j in range(profile.shape[0]):
        for k in range(profile.shape[1]):
            train_profiles_np[i, j, k] = profile[j, k]
print(train_profiles_np.shape)
print(train_profiles_np[12][12])
test_profiles_np = np.zeros((len(test_profiles), maxlen_seq, 22))
for i, profile in enumerate(test_profiles):
    for j in range(profile.shape[0]):
        for k in range(profile.shape[1]):
            test_profiles_np[i, j, k] = profile[j, k]

valid_profiles_np = np.zeros((len(valid_profiles), maxlen_seq, 22))
for i, profile in enumerate(valid_profiles):
    for j in range(profile.shape[0]):
        for k in range(profile.shape[1]):
            valid_profiles_np[i, j, k] = profile[j, k]






'''
