#####
#definitions for utility functions
#   generate triplets from data, labels, alpha

#####
import numpy as np
from scipy.spatial import distance_matrix
from scipy.io.wavfile import read as readwav


#####
#function to load speeches
#path: path to folder of TIMIT data
#label is generated from file names as they contain the subject ID
def load_data(path):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(path, f))]
    data = []
    labs = []
    for f in onlyfiles:
        data += [readwav(filename=join(path,f))[1]]
        labs += [f[:5]]
    return data, labs

#function to randomly sample segments from audio
#audios: input audio data
#labels: input label
#slength: time of sampling segments in second
#srate: sampling rate of audio
#s_samp: number of sampling segments
def sample_speeches(audios,labels,slength,srate,n_samp=10):
    X = []
    Y = []
    for series,lab in zip(audio,label):
        for _ in range(n_samp):
            if series.shape[0] < slength*srate: #if shorter than
                X += [np.pad(series,(0,int(slength*srate - series.shape[0])),'constant',constant_values=0)]
            else:
                point = random.randint(0,series.shape[0]-slength*srate)
                X += [series[point:point+slength*srate]]
            Y += [lab]
    ids = np.unique(Y).tolist()
    Y = np.array([ids.index(i) for i in Y],dtype=np.int32)
    X = np.stack(X)
    X = (X - np.tile(np.mean(X,axis=1),(X.shape[1],1)).T) / np.tile(np.std(X,axis=1),(X.shape[1],1)).T
    return np.float32(X), Y


#function to generate triplets
#   X: feature data
#   Y: label data
#   alpha: parameter of triplet loss
def gen_triplets(X,Y,alpha):
    triplets = []
    #query each label to obtain positive and negative pairs
    for lab in np.unique(Y):
        pos_idx = np.where(Y==lab)[0].tolist()
        neg_idx = np.where(Y<>lab)[0].tolist()
        Xi = X[Y==lab]
        Xi_neg = X[Y<>lab]
        ni = Xi.shape[0]
        idx = np.mgrid[0:ni,0:ni]
        ix = idx[0]
        iy = idx[1]
        l2p = np.sum((Xi[ix] - Xi[iy])**2, axis=2)
        temp = np.transpose(np.repeat(Xi,repeats=Xi_neg.shape[0]).reshape(Xi.shape[0],X.shape[1],Xi_neg.shape[0]), axes=(0,2,1))
        l2n = np.sum((temp - Xi_neg)**2, axis=2)
        for i in range(l2p.shape[0]):
            for j in range(l2p.shape[0]):
                hard_neg = np.where(l2n[i,:] <= l2p[i,j] + alpha)[0].tolist()
                triplets += [[pos_idx[i],pos_idx[j],neg_idx[k]] for k in hard_neg]
    return np.array(triplets, dtype=np.int32)


#function to generate precision recall curve
#train: training data, list of [features, true labels]
#test:  testing data, list of [features, predicted labels]
def gen_pr(train, test, k):
    trX,trY = train
    tsX,tsY = test
    distance = distance_matrix(tsX,trX)
    sorted_ind = np.argsort(distance, axis=1)
    Yr = np.repeat(tsY,repeats=distance.shape[1]).reshape(distance.shape[0],distance.shape[1])
    pred = trY[sorted_ind]
    precision = []
    recall = []
    N1 = np.bincount(trainY.astype(np.int32))
    N2 = np.bincount(testY.astype(np.int32))
    relevant = float(np.sum(N1*N2))
    for i in range(test.shape[0]):
        tp = np.sum((pred==Yr)[:,:i+1])
        retrieve = float(pred[:,:i+1].size)
        precision.append(tp/retrieve)
        recall.append(tp/relevant)
    return precision, recall
