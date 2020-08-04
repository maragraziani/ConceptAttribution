import cv2
import numpy as np
from color_analysis import *
from texture_analysis import *
import keras
import keras.backend as K
#import statsmodels.api as sm

"""
Functions:

[TEST] get_color_measure(image, mask=None, type=None, verbose=True)
[TEST] get_texture_measure(image, mask=None, type=None, verbose=True)
[TEST] get_all_color_measures(image, mask=None)
[TEST] get_all_texture_measures(image, mask=None)

get_activations(model, layer, data, labels=None, pooling=None, param_update=False, save_fold='')
[NOT WORKING ATM]

get_rcv(acts, measures, type='linear', evaluation=False, verbose=True)

predict_with_rcv maybe?

compute_mse(labels, predictions)
compute_rsquared(labels, predictions)

"""

def get_color_measure(image, mask=None, mtype=None, verbose=False):
    if mask is not None:
        print("A mask was specified")
        print("This feature has not been implemented yet")
        return None
    if mtype is None:
        print("No type was given")
        return None
    if mtype=='colorfulness':
        return colorfulness(image)
    else:
        return colorness(image, mtype, threshold=0, verbose=verbose)

def get_all_color_measures(image, mask=None, verbose=False):
    all_types = ['colorfulness',
                 'red',
                 'orange',
                 'yellow',
                 'green',
                 'cyano',
                 'blue',
                 'purple',
                 'magenta',
                 'black',
                 'white'
                ]
    cms={}
    for mtype in all_types:
        if verbose:  print(mtype)
        cms[mtype]=get_color_measure(image,mask=mask,mtype=mtype, verbose=verbose)
    return cms

def get_texture_measure(image, mask=None, mtype=None, verbose=False):
    if mask is not None:
        print("A mask was specified")
        print("This feature has been implemented in iMIMIC paper")
        return None
    if mtype is None:
        print("No type was given")
        return None
    return haralick(image, mask=mask, mtype=mtype, verbose=verbose)

def get_all_texture_measures(image, mask=None, verbose=False):
    all_types = ['dissimilarity',
                 'contrast',
                 'correlation',
                 'homogeneity',
                 'ASM',
                 'energy'
                ]
    cms={}
    for mtype in all_types:
        if verbose:  print(mtype)
        cms[mtype]=get_texture_measure(image,mask=mask,mtype=mtype)
    return cms

def binarize_image(image):
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    scaled_image = (image-min_intensity) / (max_intensity-min_intensity)
    mask = (scaled_image > 0.5)*1
    return mask

def get_region_measure(image, mask=None, mtype=None, verbose=False):
    try:
        len(mask)
    except:
        mask = binarize_image(image)
        return get_region_measure(image, mask=mask, mtype=mtype, verbose=False)
    measure_properties = skimage.measure.regionprops(mask)[0]
    return measure_properties[mtype]

def get_all_region_measures(image, mask=None, verbose=False):
    all_types=['area',
               'perimeter',
               'eccentricity',
               'orientation'
              ]
    cms={}
    for mtype in all_types:
        if verbose:    print(mtype)
        cms[mtype]=get_region_measure(image, mask=mask, mtype=mtype)
    return cms

def get_batch_activations(model, layer, batch, labels=None):
    """
    gets a keras model as input, a layer name and a batch of data
    and outputs the network activations
    """
    get_layer_output = K.function([model.layers[0].input],
                                  [model.get_layer(layer).output])
    feats = get_layer_output([batch])
    return feats[0]

def get_activations(model, layer, data, labels=None, pooling=None, param_update=False, save_fold=''):
    print("todo")
    return None

"""Support function for get_rcv"""
def linear_regression(inputs, y, random_state=1, verbose=False):
    inputs = sm.add_constant(inputs)
    model = sm.OLS(y,inputs)
    results = model.fit()
    return results

def compute_mse(labels, predictions):
    errors = labels - predictions
    sum_squared_errors = np.sum(np.asarray([pow(errors[i],2) for i in range(len(errors))]))
    mse = sum_squared_errors / len(labels)
    return mse

def compute_rsquared(labels, predictions):
    errors = labels - predictions
    sum_squared_errors = np.sum(np.asarray([pow(errors[i],2) for i in range(len(errors))]))
    # total sum of squares, TTS
    average_y = np.mean(labels)
    total_errors = labels - average_y
    total_sum_squares = np.sum(np.asarray([pow(total_errors[i],2) for i in range(len(total_errors))]))
    #rsquared is 1-RSS/TTS
    rss_over_tts =   sum_squared_errors/total_sum_squares
    rsquared = 1-rss_over_tts
    return rsquared
"""end of support functions"""

def cluster_data(inputs, mode='KMeans', n_clusters=1, random_state=1):
    if mode=='DBSCAN':
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=15, min_samples=30).fit(inputs)
    if mode=='KMeans':
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=n_clusters, random_state=random_state).fit(inputs)
    if mode=='NearestN':
        from sklear.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=n_clusters, algorithm='ball_tree').fit(indices)
        return None #not finished yet

def local_linear_regression(inputs, y, max_clusters, random_state=1, verbose=False):
    # find clusters
    # solve regression in the clusters
    # returns a
    clustering = cluster_data(inputs, mode='KMeans', n_clusters=max_clusters, random_state=random_state)
    if verbose:
        print(clustering.labels_)
    return clustering, clustering.labels_

def get_rcv(acts, measures, type='global linear', max_clusters=1, evaluation=True, random_state=1, verbose=True):
    """"
    Returns the RCV
    """
    if type=='global linear':
        #rcv_result = linear_regression(acts, measures, random_state=random_state, verbose=True)
        if evaluation:
            from sklearn.model_selection import train_test_split
            train_acts, test_acts, train_meas, test_meas = train_test_split(acts,
                                                                            measures,
                                                                            test_size=0.3,
                                                                            random_state=random_state)
            rcv_result = linear_regression(train_acts, train_meas, random_state=random_state, verbose=True)
            rsquared = rcv_result.rsquared
            test_data_ = sm.add_constant(test_acts)
            predictions = rcv_result.predict(test_data_)
            mse_test = compute_mse(test_meas, predictions)
            r2_test = compute_rsquared(test_meas, predictions)
        #rcv_result = linear_regression(acts, measures, random_state=random_state, verbose=True)
        if verbose:
            print("Global linear regression under euclidean assumption")
            print("Random state: ", random_state)
            print("R2: ", rcv_result.rsquared)
            #if evaluation:
            #    print("MSE: ", mse)
        print("TEST mse: {}, r2: {}".format(mse_test, r2_test))
            #print(rcv_result.summary())
    elif type=='local linear':
        if verbose:
            print("Local linear regression under Euclidean assumption")
        if evaluation:
            local_rcvs = {}
            from sklearn.model_selection import train_test_split
            train_acts, test_acts, train_meas, test_meas = train_test_split(acts,
                                                                            measures,
                                                                            test_size=0.3,
                                                                            random_state=random_state)
            #import pdb; pdb.set_trace()
            clusterer, clustering_labels = local_linear_regression(train_acts, train_meas, max_clusters)
            n_clusters = np.max(clustering_labels) + 1 # we start from 0 label
            for cluster_id in range(n_clusters):
                datapoint_idxs = np.argwhere(clustering_labels==cluster_id).T[0]
                rcv_result = linear_regression(train_acts[datapoint_idxs], train_meas[datapoint_idxs],random_state=random_state, verbose=True)
                rsquared = rcv_result.rsquared
                local_rcvs[cluster_id] = rcv_result
                if verbose:
                    print("Cluster no. {}".format(cluster_id))
                    print("Random state: ", random_state)
                    print("R2: ", rcv_result.rsquared)
            #(n,d)=test_acts.shape
            #centroids=np.zeros((n*n_clusters, d))
            #test_acts_duplicates=np.zeros((n*n_clusters,d))
            #for cluster_id in range(n_clusters):
            #    centroids[cluster_id*n:cluster_id*n+n]=cluster_centers[cluster_id]
            #    test_acts_duplicates[cluster_id*n:cluster_id*n+n]=test_acts
            #import pdb; pdb.set_trace()
            #distance_from_centroids = np.sum(np.power(test_acts_duplicates - centroids)
            avg_mse = 0
            avg_r2 = 0
            close_clusters=clusterer.predict(test_acts)
            for cluster_id in range(n_clusters):
                data_idxs = np.argwhere(close_clusters==cluster_id).T[0]
                test_data_ = sm.add_constant(test_acts[data_idxs])
                predictions = local_rcvs[cluster_id].predict(test_data_)
                mse = compute_mse(test_meas[data_idxs], predictions)
                r2 = compute_rsquared(test_meas[data_idxs], predictions)
                if verbose:
                    print("TEST cluster id: {}, mse: {}, r2: {}".format(cluster_id, mse, r2))
                avg_mse+=mse
                avg_r2+=r2
            avg_r2/=n_clusters
            avg_mse/=n_clusters
            print("Cumulative MSE: {}, Avg R2: {}".format(avg_mse, avg_r2))

            #import pdb; pdb.set_trace()

               #print(rcv_result.summary())
            return train_acts, clustering_labels, avg_mse, avg_r2
    elif type =='local UMAP':
        if verbose:
            print("Local linear regression on UMAP clustering with euclidean distances (UMAP)")
        import umap
        local_rcvs = {}
        from sklearn.model_selection import train_test_split
        train_acts, test_acts, train_meas, test_meas = train_test_split(acts,
                                                                        measures,
                                                                        test_size=0.3,
                                                                        random_state=random_state)
        transform = umap.UMAP(n_neighbors=15,#15
                     min_dist=0.3,
                     random_state=random_state,
                     metric='euclidean').fit(train_acts)
        train_embedding=transform.embedding_

        clusterer, clustering_labels = local_linear_regression(train_embedding, train_meas, max_clusters)

        n_clusters = np.max(clustering_labels) + 1 # we start from 0 label
        for cluster_id in range(n_clusters):
            datapoint_idxs = np.argwhere(clustering_labels==cluster_id).T[0]
            rcv_result = linear_regression(train_acts[datapoint_idxs], train_meas[datapoint_idxs],random_state=random_state, verbose=True)
            #rcv_result = linear_regression(train_embedding[datapoint_idxs], train_meas[datapoint_idxs],random_state=random_state, verbose=True)
            rsquared = rcv_result.rsquared
            local_rcvs[cluster_id] = rcv_result
            if verbose:
                print("Cluster no. {}".format(cluster_id))
                print("Random state: ", random_state)
                print("R2: ", rcv_result.rsquared)
            avg_mse = 0
            avg_r2 = 0

        test_embedding = transform.transform(test_acts)
        close_clusters=clusterer.predict(test_embedding)
        for cluster_id in range(n_clusters):
            data_idxs = np.argwhere(close_clusters==cluster_id).T[0]
            test_data_ = sm.add_constant(test_acts[data_idxs])
            #test_data_ = sm.add_constant(test_embedding[data_idxs])
            predictions = local_rcvs[cluster_id].predict(test_data_)
            mse = compute_mse(test_meas[data_idxs], predictions)
            r2 = compute_rsquared(test_meas[data_idxs], predictions)
            if verbose:
                print("TEST cluster id: {}, mse: {}, r2: {}".format(cluster_id, mse, r2))
            avg_mse+=mse
            avg_r2+=r2
        avg_r2/=n_clusters
        avg_mse/=n_clusters
        print("Cumulative MSE: {}, Avg R2: {}".format(avg_mse, avg_r2))

        return transform, clusterer, train_acts, clustering_labels, avg_mse, avg_r2, local_rcvs #, train_meas

    elif type=='global manifold':
        if verbose:
            print("Global linear regression on unknown manifold")
    return
