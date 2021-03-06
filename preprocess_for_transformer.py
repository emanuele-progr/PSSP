''' Handling the data io '''
import os
import argparse
import torch
import pickle
import numpy as np


PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

TRAIN_PATH = './pssp-data/cullpdb+profile_6133_filtered.npy.gz'
TEST_PATH = './pssp-data/cb513+profile_split1.npy.gz'

TRAIN_URL = "http://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133_filtered.npy.gz"
TEST_URL = "http://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz"

def load_gz(path):  # load a .npy.gz file
    if path.endswith(".gz"):
        #f = gzip.open(path, 'rb')
        return np.load(path)
    else:
        return np.load(path)

def save_text(data, save_path):
    with open(save_path, mode='w') as f:
        f.write('\n'.join(data))


def save_picke(data, save_path):
    with open(save_path, mode="wb") as f:
        pickle.dump(data, f)

def AA_PATH(key): return f'./pssp-data/aa_{key}.txt'


def SP_PATH(key): return f'./pssp-data/sp_{key}.pkl'


def PSS_PATH(key): return f'./pssp-data/pss_{key}.txt'


TRAIN_PATH = 'data/cullpdb+profile_6133_filtered.npy.gz'
TEST_PATH = 'data/cb513+profile_split1.npy.gz'
##### TRAIN DATA #####


def download_dataset():
    print('[Info] Downloading CB513 dataset ...')
    if not (os.path.isfile(TRAIN_PATH) and os.path.isfile(TEST_PATH)):
        os.makedirs('./pssp-data', exist_ok=True)
        os.system(f'wget -O {TRAIN_PATH} {TRAIN_URL}')
        os.system(f'wget -O {TEST_PATH} {TEST_URL}')


def make_datasets():
    print('[Info] Making datasets ...')

    # train dataset
    X_train, y_train, seq_len_train = make_dataset(TRAIN_PATH)
    make_dataset_for_transformer(X_train, y_train, seq_len_train, 'train')

    # test dataset
    X_test, y_test, seq_len_test = make_dataset(TEST_PATH)
    make_dataset_for_transformer(X_test, y_test, seq_len_test, 'test')


def make_dataset(path):
    data = load_gz(path)
    data = data.reshape(-1, 700, 57)

    idx = np.append(np.arange(21), np.arange(35, 56))
    X = data[:, :, idx]
    X = X.transpose(0, 2, 1)
    X = X.astype('float32')

    y = data[:, :, 22:30]
    y = np.array([np.dot(yi, np.arange(8)) for yi in y])
    y = y.astype('float32')

    mask = data[:, :, 30] * -1 + 1
    seq_len = mask.sum(axis=1)
    seq_len = seq_len.astype(int)

    return X, y, seq_len


def make_dataset_for_transformer(X, y, seq_len, key):
    X_amino = X[:, :21, :]
    X_profile = X[:, 21:, :]

    amino_acid_array = get_amino_acid_array(X_amino, seq_len)
    save_path = AA_PATH(key)
    save_text(amino_acid_array, save_path)
    print(f'[Info] Saved amino_acid_array for {key} in {save_path}')

    seq_profile = get_seq_profile(X_profile, seq_len)
    save_path = SP_PATH(key)
    save_picke(seq_profile, save_path)
    print(f'[Info] Saved seq_profile for {key} in {save_path}')

    pss_array = get_pss_array(y, seq_len)
    save_path = PSS_PATH(key)
    save_text(pss_array, save_path)
    print(f'[Info] Saved pss_array for {key} in {save_path}')


def get_amino_acid_array(X_amino, seq_len):
    amino_acid = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M',
                  'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X']
    amino_acid_array = []
    for X, l in zip(X_amino, seq_len):
        acid = {}
        for i, aa in enumerate(amino_acid):
            keys = np.where(X[i] == 1)[0]
            values = [aa] * len(keys)
            acid.update(zip(keys, values))
        aa_str = ' '.join([acid[i] for i in range(l)])

        amino_acid_array.append(aa_str)
    return amino_acid_array


def get_pss_array(label, seq_len):
    pss_icon = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
    pss_array = []
    for target, l in zip(label, seq_len):
        pss = np.array(['Nofill'] * l)
        target = target[:l]
        for i, p in enumerate(pss_icon):
            idx = np.where(target == i)[0]
            pss[idx] = p

        pss_str = ' '.join([pss[i] for i in range(l)])
        pss_array.append(pss_str)

    return pss_array


def get_seq_profile(X_profile, seq_len):
    seq_profile = []
    for sp, l in zip(X_profile, seq_len):
        seq_profile.append(sp[:, :l])
    return seq_profile


def read_instances_from_file(inst_file, max_sent_len, keep_case, without_bos_eos=False, bigrams=False):
    ''' Convert file into word seq lists and vocab '''

    word_insts = []
    trimmed_sent_count = 0
    with open(inst_file) as f:
        for sent in f:
            if not keep_case:
                sent = sent.lower()
            words = sent.split()
            if bigrams:
                res = []
                for i in range(0, len(words)):
                    res.append(words[i - 1] + words[i])
                words = res

            if len(words) > max_sent_len:
                trimmed_sent_count += 1
            word_inst = words[:max_sent_len]

            if word_inst:
                if without_bos_eos:
                    word_insts += [word_inst]
                else:
                    word_insts += [[BOS_WORD] + word_inst + [EOS_WORD]]
            else:
                word_insts += [None]

    print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return word_insts


def build_vocab_idx(word_insts, min_word_count, without_bos_eos=False):
    ''' Trim vocab by number of occurence '''

    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        BOS_WORD: BOS,
        EOS_WORD: EOS,
        PAD_WORD: PAD,
        UNK_WORD: UNK
    }
    if without_bos_eos:
        word2idx = {
            PAD_WORD: PAD,
            UNK_WORD: UNK
        }

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                # print(word, len(word2idx))
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx


def convert_instance_to_idx_seq(word_insts, word2idx):
    test = [[word2idx.get(w, UNK) for w in s] for s in word_insts]
    return test


def load_picke_data(path):
    with open(path, mode="rb") as f:
        data = pickle.load(f)
    return data


def main():
    ''' Main function '''
    if not os.path.exists('./data'):
        os.makedirs('./data')
    download_dataset()
    if not os.path.exists('./pssp-data'):
        os.makedirs('./pssp-data', exist_ok=True)
    make_datasets()

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', default='./pssp-data/aa_train.txt')
    parser.add_argument('-train_tgt', default='./pssp-data/pss_train.txt')
    parser.add_argument('-train_sp', default='./pssp-data/sp_train.pkl')

    parser.add_argument('-valid_src', default='./pssp-data/aa_test.txt')
    parser.add_argument('-valid_tgt', default='./pssp-data/pss_test.txt')
    parser.add_argument('-valid_sp', default='./pssp-data/sp_test.pkl')

    parser.add_argument('-save_data', default='./pssp-data/data.pt')
    parser.add_argument('-max_len', '--max_word_seq_len',
                        type=int, default=700)
    parser.add_argument('-min_word_count', type=int, default=1)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)
    parser.add_argument('-without_bos_eos', action='store_true')
    parser.add_argument('-src_bigrams', default=False)
    parser.add_argument('-tgt_bigrams', default=False)

    opt = parser.parse_args()
    opt.max_token_seq_len = opt.max_word_seq_len + 2  # include the <s> and </s>

    src_n = 1
    '''if opt.src_bigrams:
        src_n = 2'''

    tgt_n = 1
    '''if opt.tgt_bigrams:
        tgt_n = 2'''
    # Training set
    train_src_word_insts = read_instances_from_file(
        opt.train_src, int(opt.max_word_seq_len / src_n), opt.keep_case, opt.without_bos_eos, opt.src_bigrams)
    train_tgt_word_insts = read_instances_from_file(
        opt.train_tgt, int(opt.max_word_seq_len / tgt_n), opt.keep_case, opt.without_bos_eos, opt.tgt_bigrams)

    if len(train_src_word_insts) != len(train_tgt_word_insts):
        print('[Warning] The training instance count is not equal.')
        min_inst_count = min(len(train_src_word_insts),
                             len(train_tgt_word_insts))
        train_src_word_insts = train_src_word_insts[:min_inst_count]
        train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]

    # - Remove empty instances
    train_src_word_insts, train_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(train_src_word_insts, train_tgt_word_insts) if s and t]))



    # print(train_src_word_insts)

    # Validation set
    valid_src_word_insts = read_instances_from_file(
        opt.valid_src, opt.max_word_seq_len, opt.keep_case, opt.without_bos_eos)
    valid_tgt_word_insts = read_instances_from_file(
        opt.valid_tgt, opt.max_word_seq_len, opt.keep_case, opt.without_bos_eos)

    if len(valid_src_word_insts) != len(valid_tgt_word_insts):
        print('[Warning] The validation instance count is not equal.')
        min_inst_count = min(len(valid_src_word_insts),
                             len(valid_tgt_word_insts))
        valid_src_word_insts = valid_src_word_insts[:min_inst_count]
        valid_tgt_word_insts = valid_tgt_word_insts[:min_inst_count]

    # - Remove empty instances
    valid_src_word_insts, valid_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(valid_src_word_insts, valid_tgt_word_insts) if s and t]))

    # Build vocabulary
    if opt.vocab:
        predefined_data = torch.load(opt.vocab)
        assert 'dict' in predefined_data

        print('[Info] Pre-defined vocabulary found.')
        src_word2idx = predefined_data['dict']['src']
        tgt_word2idx = predefined_data['dict']['tgt']
    else:
        if opt.share_vocab:
            print('[Info] Build shared vocabulary for source and target.')
            word2idx = build_vocab_idx(
                train_src_word_insts + train_tgt_word_insts, opt.min_word_count, opt.without_bos_eos)
            src_word2idx = tgt_word2idx = word2idx

        else:
            print('[Info] Build vocabulary for source.')
            src_word2idx = build_vocab_idx(
                train_src_word_insts, opt.min_word_count, opt.without_bos_eos)
            print(src_word2idx)
            print('[Info] Build vocabulary for target.')
            tgt_word2idx = build_vocab_idx(
                train_tgt_word_insts, opt.min_word_count, opt.without_bos_eos)
            print(tgt_word2idx)

    # word to index
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(
        train_src_word_insts, src_word2idx)
    valid_src_insts = convert_instance_to_idx_seq(
        valid_src_word_insts, src_word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_instance_to_idx_seq(
        train_tgt_word_insts, tgt_word2idx)
    valid_tgt_insts = convert_instance_to_idx_seq(
        valid_tgt_word_insts, tgt_word2idx)


    # print(train_src_insts)
    # read sequences profile
    train_seq_profile = load_picke_data(opt.train_sp)
    valid_seq_profile = load_picke_data(opt.valid_sp)


    data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src': train_src_insts,
            'sp': train_seq_profile,
            'tgt': train_tgt_insts},
        'valid': {
            'src': valid_src_insts,
            'sp': valid_seq_profile,
            'tgt': valid_tgt_insts}}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')


if __name__ == '__main__':
    main()