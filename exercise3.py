import os
import shlex
import argparse
from tqdm import tqdm

# for python3: read in python2 pickled files
import _pickle as cPickle

import gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
import numpy as np
import cv2
from parmap import parmap

# parmap can only accept one parameter & we use fork instad of spawn
from functools import partial
import multiprocessing
multiprocessing.set_start_method("fork", force=True)


# esvm_loop: at first in esvm, here for multiprocessing
def esvm_loop(i, encs_test, encs_train, C):

    # ith test exemplar
    enc_test_i = encs_test[i]

    # positive sample + all negative samples (train set)
    X = np.vstack([enc_test_i.reshape(1,-1), encs_train])
    # create lables: first is positive, all others negative
    y = np.array([1] + [0]*len(encs_train))

    # create linear SVM
    svm = LinearSVC(C=C, class_weight='balanced',
                    max_iter=10000, random_state=42)
    # train SVM
    svm.fit(X, y)

    # make weight vector one dimensional
    weights = svm.coef_[0].flatten()
    # calculate norm
    norm = np.linalg.norm(weights)
    # L2 norm, for ignoring the scale of the weights
    if norm > 0:
        weights = weights / norm

    # return weight vectore
    return weights.reshape(1, -1)


def parseArgs(parser):
    parser.add_argument('--labels_test', 
                        help='contains test images/descriptors to load + labels')
    parser.add_argument('--labels_train', 
                        help='contains training images/descriptors to load + labels')
    parser.add_argument('-s', '--suffix',
                        default='_SIFT_patch_pr.pkl.gz',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--in_test',
                        help='the input folder of the test images / features')
    parser.add_argument('--in_train',
                        help='the input folder of the training images / features')
    parser.add_argument('--overwrite', action='store_true',
                        help='do not load pre-computed encodings')
    parser.add_argument('--powernorm', action='store_true',
                        help='use powernorm')
    parser.add_argument('--gmp', action='store_true',
                        help='use generalized max pooling')
    parser.add_argument('--gamma', default=1, type=float,
                        help='regularization parameter of GMP')
    parser.add_argument('--C', default=1000, type=float, 
                        help='C parameter of the SVM')
    return parser

def getFiles(folder, pattern, labelfile):
    """ 
    returns files and associated labels by reading the labelfile 
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels 
    """
    # read labelfile
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()
    
    # get filenames from labelfile
    all_files = []
    labels = []
    check = True
    for line in all_lines:
        # using shlex we also allow spaces in filenames when escaped w. ""
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # strip all known endings, note: os.path.splitext() doesnt work for
        # '.' in the filenames, so let's do it this way...
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb','.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p,'')

        # get now new file name
        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels

def loadRandomDescriptors(files, max_descriptors):
    """ 
    load roughly `max_descriptors` random descriptors
    parameters:
        files: list of filenames containing local features of dimension D
        max_descriptors: maximum number of descriptors (Q)
    returns: QxD matrix of descriptors
    """
    # let's just take 100 files to speed-up the process
    max_files = 100
    indices = np.random.permutation(max_files)
    files = np.array(files)[indices]
   
    # rough number of descriptors per file that we have to load
    max_descs_per_file = int(max_descriptors / len(files))

    descriptors = []
    for i in tqdm(range(len(files))):
        with gzip.open(files[i], 'rb') as ff:
            # for python2
            # desc = cPickle.load(ff)
            # for python3
            desc = cPickle.load(ff, encoding='latin1')
            
        # get some random ones
        indices = np.random.choice(len(desc),
                                   min(len(desc),
                                       int(max_descs_per_file)),
                                   replace=False)
        desc = desc[ indices ]
        descriptors.append(desc)
    
    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors

#cn_1. codebook generation
def dictionary(descriptors, n_clusters):
    """ 
    return cluster centers for the descriptors 
    parameters:
        descriptors: NxD matrix of local descriptors
        n_clusters: number of clusters = K
    returns: KxD matrix of K clusters
    """
    # TODO
    # 1000 features until convergence
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, 
                            batch_size=1000, 
                            random_state=42)
    # fit k-means
    kmeans.fit(descriptors)
    clusters = kmeans.cluster_centers_
    
    # dummy = np.array([42])
    # return dummy
    return clusters  # return centers of clusters


def assignments(descriptors, clusters):
    """ 
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix
        clusters: KxD cluster matrix
    returns: TxK assignment matrix
    """
    # compute nearest neighbors
    # TODO
    
    # calculate distances between descriptors and cluster centers
    # descriptors: (T, D), clusters: (K, D)
    distances = np.linalg.norm(
        descriptors[:, np.newaxis, :] - clusters[np.newaxis, :, :], 
        axis=2  # calculate norm along the last axis
    )
    # numpy boardcast and auto-alignment,here axis=2 means calculate Euclidean distance
    
    # find the nearest cluster center for each descriptor
    # axis=1 means find the index of the minimum value along each row
    nearest_indices = np.argmin(distances, axis=1) 
    
    # create hard assignment
    assignment = np.zeros((len(descriptors), len(clusters)))
    # TODO

    # hard assignment: set the corresponding cluster index to 1
    assignment[np.arange(len(descriptors)), nearest_indices] = 1

    return assignment

def vlad(files, mus, powernorm, gmp=False, gamma=1000):
    """
    compute VLAD encoding for each files
    parameters: 
        files: list of N files containing each T local descriptors of dimension
        D
        mus: KxD matrix of cluster centers
        gmp: if set to True use generalized max pooling instead of sum pooling
    returns: NxK*D matrix of encodings
    """
    K = mus.shape[0]
    encodings = []

    for f in tqdm(files):
        with gzip.open(f, 'rb') as ff:
            desc = cPickle.load(ff, encoding='latin1')
        a = assignments(desc, mus)
        
        T,D = desc.shape
        f_enc = np.zeros((D*K), dtype=np.float32)
        for k in range(mus.shape[0]):
            # it's faster to select only those descriptors that have
            # this cluster as nearest neighbor and then compute the 
            # difference to the cluster center than computing the differences
            # first and then select
            # TODO 
            
            # find indices of descriptors assigned to cluster k
            indices = np.where(a[:, k] == 1)[0]
            
            if len(indices) > 0:
                # calculate residuals：desc[indices] - mus[k]
                residuals = desc[indices] - mus[k]
                
                # sum: axis = 0 means summing over all selected descriptors
                sum_residuals = np.sum(residuals, axis=0)
                
                # store in the corresponding position of f_enc
                f_enc[k*D:(k+1)*D] = sum_residuals
    
        # c) power normalization
        if powernorm:
            # TODO
            
            # sign * sqrt(abs(x)):scaling features
            f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))
        
        # l2 normalization
        # TODO
        
        # l2 norm:square root of sum of squares of features
        norm = np.linalg.norm(f_enc)
        if norm > 0:
            f_enc = f_enc / norm
        
        encodings.append(f_enc)
    
    return encodings


def esvm(encs_test, encs_train, C=1000):

    def loop(i):
        return esvm_loop(i, encs_test, encs_train, C)
    # tqdm is used for showing progress bar
    # calculate weight vector of every test sample, make it as a list
    new_encs = list(parmap(loop, tqdm(range(len(encs_test)))))
    #make list as a matrix to calculate
    new_encs = np.concatenate(new_encs, axis=0)
    return new_encs


# calculate similarity between samples
def distances(encs):
    """ 
    compute pairwise distances 

    parameters:
        encs:  TxK*D encoding matrix
    returns: TxT distance matrix
    """
    # compute cosine distance = 1 - dot product between l2-normalized
    # encodings
    # TODO
    
    # 1. l2 normalization
    encs_norm = encs / np.linalg.norm(encs, axis=1, keepdims=True)
    
    # 2. calculate cosine similarity
    cosine_sim = np.dot(encs_norm, encs_norm.T)
    
    # 3. distance = 1 - cosine similarity
    dists = 1 - cosine_sim
    
    # mask out distance with itself to avoid make itself Top-1
    np.fill_diagonal(dists, np.finfo(dists.dtype).max)
    return dists

def evaluate(encs, labels):
    """
    evaluate encodings assuming using associated labels
    parameters:
        encs: TxK*D encoding matrix
        labels: array/list of T labels
    """
    dist_matrix = distances(encs)
    # sort each row of the distance matrix
    indices = dist_matrix.argsort()

    n_encs = len(encs)

    mAP = []
    correct = 0
    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs-1):
            if labels[ indices[r,k] ] == labels[ r ]:
                rel += 1
                precisions.append( rel / float(k+1) )
                if k == 0:
                    correct += 1
        avg_precision = np.mean(precisions)
        mAP.append(avg_precision)
    mAP = np.mean(mAP)

    print('Top-1 accuracy: {} - mAP: {}'.format(float(correct) / n_encs, mAP))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('retrieval')
    parser = parseArgs(parser)
    args = parser.parse_args()
    np.random.seed(42)  # fix random seed
    
    files_train, labels_train = getFiles(args.in_train, args.suffix,
                                         args.labels_train)
    print('#train: {}'.format(len(files_train)))
    files_test, labels_test = getFiles(args.in_test, args.suffix,
                                       args.labels_test)
    print('#test: {}\n'.format(len(files_test)))
    
    # a) dictionary
    print('=== Task a: Codebook Generation ===')
    if not os.path.exists('mus.pkl.gz') or args.overwrite:
        # load random descriptors from training set
        descriptors = loadRandomDescriptors(files_train, max_descriptors=500000)
        print('> loaded {} descriptors'.format(len(descriptors)))
        
        # code generation with k-means
        print('> compute dictionary with k-means (k=100)')
        mus = dictionary(descriptors, n_clusters=100)
        
        with gzip.open('mus.pkl.gz', 'wb') as fOut:
            cPickle.dump(mus, fOut, -1)
        print('> dictionary saved to mus.pkl.gz\n')
    else:
        with gzip.open('mus.pkl.gz', 'rb') as f:
            mus = cPickle.load(f)
        print('> loaded existing dictionary from mus.pkl.gz\n')

    # b) VLAD encoding
    print('> compute VLAD for test')
    files_test, labels_test = getFiles(args.in_test, args.suffix,
                                    args.labels_test)
    print('#test: {}'.format(len(files_test)))
    print('> evaluate RAW VLAD (no normalization)')
    enc_test_raw = vlad(files_test, mus, powernorm=False, gmp=False, gamma=1.0)
    

    print('>>> Results for Task b (vlad without normalization):')
    evaluate(enc_test_raw, labels_test)
    print()
    
    # c) VLAD encoding with normalization
    print('=== Task c: VLAD Encoding (with normalization) ===')
    print('> compute NORMALIZED VLAD for test set')
    enc_test_norm = vlad(files_test, mus, args.powernorm, args.gmp, args.gamma)
    
    # save normalized VLAD
    fname_norm = 'enc_test_gmp{}.pkl.gz'.format(args.gamma) if args.gmp else 'enc_test.pkl.gz'
    with gzip.open(fname_norm, 'wb') as fOut:
        cPickle.dump(enc_test_norm, fOut, -1)
    
    print('> evaluate NORMALIZED VLAD (Task c requirement)')
    print('>>> Results for Task c:')
    evaluate(enc_test_norm, labels_test)
    
    # compare outcomes of Task b and c
    print('\n>>> Comparison Task b vs Task c:')
    print('    (Run separate evaluation to get mAP values for comparison)')
    print('    Expected: Normalization should improve mAP significantly\n')
    
    # prepare training set encodings for ESVM
    print('> compute NORMALIZED VLAD for train set (for E-SVM)')
    enc_train_norm = vlad(files_train, mus, args.powernorm, args.gmp, args.gamma)
    
    # save normalized VLAD for train set
    fname_train = 'enc_train_gmp{}.pkl.gz'.format(args.gamma) if args.gmp else 'enc_train.pkl.gz'
    with gzip.open(fname_train, 'wb') as fOut:
        cPickle.dump(enc_train_norm, fOut, -1)
    
    # d) Exemplar-SVM
    print('\n=== Task d: Exemplar Classification ===')
    print('> compute Exemplar-SVM for each test sample')
    print('> using C = {}, class_weight = balanced'.format(args.C))
    
    # calculate exemplar SVM encodings for test set
    enc_test_esvm = esvm(enc_test_norm, enc_train_norm, args.C)
    
    print('> evaluate with Exemplar-SVM (Task d requirement)')
    print('>>> Results for Task d:')
    evaluate(enc_test_esvm, labels_test)
    
    # Summary
    print('\n' + '='*60)
    print('SUMMARY OF ALL TASKS')
    print('='*60)
    print('Dataset:')
    print('  - Training samples: {}'.format(len(files_train)))
    print('  - Test samples: {}'.format(len(files_test)))
    print()
    print('To compare results (mAP values needed for each task):')
    print('  1. Task b (Raw VLAD):            Run without --powernorm')
    print('  2. Task c (Normalized VLAD):     Run with --powernorm')
    print('  3. Task d (Exemplar-SVM):        Run with --powernorm and --C 1000')
    print()
    print('Note: mAP is the required evaluation metric for all tasks.')
    print('Top-1 accuracy is shown as additional information.')
    print('='*60)

