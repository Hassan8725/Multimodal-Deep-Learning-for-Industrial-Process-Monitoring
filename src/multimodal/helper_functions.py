import numpy as np
import os
import random
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer



def reproducible_comp():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

# read config files of accelerometer data, return rate in Hz and scale in g
def read_config(file):
    # open files
    with open(file, "r") as f:
        # read lines in file
        lines = f.readlines()
    # split data by ","
    _, _, rate, scale = lines[0].split(",")
    # get rid of " g" and " Hz" and get values
    rate = float(rate.split(":")[-1][:-2])
    scale = float(scale.split(":")[-1][:-2])
    # return rate and scale values
    return rate, scale


# Defining the Grad-CAM algorithm
def grad_cam(layer_name, data):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    last_conv_layer_output, preds = grad_model(data)
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(data)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
        
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    pooled_grads = tf.reduce_mean(grads, axis=(0))
    
    last_conv_layer_output = last_conv_layer_output[0]
    
    heatmap = last_conv_layer_output * pooled_grads
    heatmap = tf.reduce_mean(heatmap, axis=(1))
    heatmap = np.expand_dims(heatmap,0)
    return heatmap



# find threshold and take norm of x, y, z peaks, float determines multiples of std (here 4 * standard deviation)
def detect_peaks(acc_data, threshold_scale: float = 4, cluster_min_distance=10):
    # take norm of x, y, z values
    norm = np.linalg.norm(acc_data[["x", "y", "z"]].values, axis=1)
    # take convolution of norm
    norm = np.convolve(norm, np.ones(5) / 5, mode='same')
    # set threshold to standard deviation of norm times (in this case 4)
    threshold = norm.std() * threshold_scale
    # return largest norm if greater than threshold
    return [np.argmax(norm)] if norm.max() > threshold else []

def get_acc_force_list(loaded_data):
    # list to cache data
    tmpacclist = []
    tmpfnlist = []
    segment_size = 500
    # go through all rows in dl dataframe
    for (acc_present, cfg, acc_data, acc_id), (fn_present, fn_data) in loaded_data:
        # if data is available for acc and fn
        if acc_present and fn_present:
            # get meta data
            rate, scale, accId, fnId, die_size, position, category, lubrication, slug_pos = cfg

            # (Acceleration data)
            # find peaks in data with detect_peaks in acceleration data
            peaks = detect_peaks(acc_data, 9, 10)
            # go through all peaks
            if len(peaks)!=0:
                 # (Force Displacement Curves data)
            # get fn data for each run and set length to 118 due to NaN values 
                tmpfn0 = fn_data[:119, 1]
                #write fn data in list
                tmpfnlist.append((cfg[2],cfg[3],cfg[6],tmpfn0))
                
                for i_peak, peak in enumerate(peaks):
                    # check if window of peak is in bounds
                    if ((len(acc_data) - peak) + segment_size // 2) < len(acc_data) and (peak - segment_size // 2) > 0:
                        # get window of peak with size of segment_size
                        tmpacc = acc_data.iloc[max(0, peak - segment_size // 2): peak + segment_size // 2]
                # append category, x, y, z values to intermediary list for each row
                tmpacclist.append((cfg[2],cfg[3],cfg[6], tmpacc["x"].values, tmpacc["y"].values, tmpacc["z"].values))
    return tmpfnlist, tmpacclist


def get_final_df(inputlist, is_acc_data, is_multilabel):
    # transform list to dataframe and rename columns
    tmpdf = pd.DataFrame(inputlist)
    if is_acc_data==1:
        tmpdf= tmpdf[[2,3,4,5]]
        tmpdf.columns = ['category', 'x', 'y', 'z']
    else:
        tmpdf= tmpdf[[2,3]]
        tmpdf.columns = ['category', 'fn']

    # exclude unnecessary categories from preliminary tests or errors
    tmpdf.drop(tmpdf[tmpdf['category'] == "Overrun"].index, inplace=True)
    tmpdf.drop(tmpdf[tmpdf['category'] == "no fn data"].index, inplace=True)
    tmpdf.drop(tmpdf[tmpdf['category'] == "Referenz3"].index, inplace=True)

    '''
        # FOR MULTICLASS: exclude 3/4 of ref values and 100/350 angle values due to uneven distribution
        '''
    tmpdf_ref = tmpdf.loc[(tmpdf['category'] == "Referenz")]
    tmpdf_ref = tmpdf_ref.iloc[::8]
    tmpdf_angle = tmpdf.loc[(tmpdf['category'] == "angle")]
    tmpdf_angle = tmpdf_angle.iloc[100:]
    tmpdf_refangle = pd.concat([tmpdf_ref, tmpdf_angle])
    tmpdf_other1 = tmpdf[tmpdf['category'] != 'Referenz']
    tmpdf_other = tmpdf_other1[tmpdf_other1['category'] != 'angle']
    tmpdf = pd.concat([tmpdf_other, tmpdf_refangle])

    if is_multilabel == 0:
        '''
        # FOR MULTICLASS: assign "ref" with 0 and "fault" with 1,2,3,4,5,6
        '''
        tmpdf.loc[tmpdf['category'] == 'Referenz', 'category'] = 0 
        tmpdf.loc[tmpdf['category'] == 'Dünn', 'category'] = 1
        tmpdf.loc[tmpdf['category'] == 'Slug', 'category'] = 2
        tmpdf.loc[tmpdf['category'] == 'Worn', 'category'] = 3
        tmpdf.loc[tmpdf['category'] == 'Kombi Worn+Dünn', 'category'] = 4
        tmpdf.loc[tmpdf['category'] == 'Kombi Slug+Dünn', 'category'] = 5
        tmpdf.loc[tmpdf['category'] == 'Kombi Worn+Slug', 'category'] = 6

        if is_acc_data==1:
            # acc split dataframe in label (category; str) and values (x,y,z; list with 500 values per column)
            data_df = tmpdf.iloc[:, 1:4].copy()

            features = np.array(data_df)
            category = tmpdf["category"]
        else:
            # append 17 zeros in front of the measurement of classes "Referenz"= 0, "Dünn"=1, "Slug"=2, "Kombi Slug+Dünn"=5 due to shorter stamp
            for counter in range(0, len(tmpdf)): # loop over all measurements
                if tmpdf.iloc[counter, 0] in [0,1,2,5]:
                    tmpdf.iloc[counter, 1] = np.insert(tmpdf.iloc[counter, 1], 0, [0]*17)
            #fn split dataframe in label (category; str) and values (f; list with 118 values per column)
            data_fn1 = tmpdf.iloc[:,1].copy()
            data_df = pd.DataFrame(data_fn1)
            features = np.array(data_df)
            category = tmpdf["category"]
        
        category_names = ['Referenz','Dünn','Slug','Worn','Kombi Worn+Dünn','Kombi Slug+Dünn','Kombi Worn+Slug']

    if is_multilabel == 1:
            '''
            # FOR Multi Label: 
            '''
            tmpdf.loc[tmpdf['category'] == 'Kombi Worn+Dünn', 'category'] = 'Worn+Dünn'
            tmpdf.loc[tmpdf['category'] == 'Kombi Slug+Dünn', 'category'] = 'Slug+Dünn'
            tmpdf.loc[tmpdf['category'] == 'Kombi Worn+Slug', 'category'] = 'Worn+Slug'

            if is_acc_data==1:
                # acc split dataframe in label (category; str) and values (x,y,z; list with 500 values per column)
                data_df = tmpdf.iloc[:, 1:4].copy()

                features = np.array(data_df)
                category = tmpdf["category"]
            else:
                # append 17 zeros in front of the measurement of classes "Referenz"= 0, "Dünn"=1, "Slug"=2, "Kombi Slug+Dünn"=5 due to shorter stamp
                for counter in range(0, len(tmpdf)): # loop over all measurements
                    if tmpdf.iloc[counter, 0] in [0,1,2,5]:
                        tmpdf.iloc[counter, 1] = np.insert(tmpdf.iloc[counter, 1], 0, [0]*17)
                #fn split dataframe in label (category; str) and values (f; list with 118 values per column)
                data_fn1 = tmpdf.iloc[:,1].copy()
                data_df = pd.DataFrame(data_fn1)
                features = np.array(data_df)
                category = tmpdf["category"]
            
            category = category.str.split('+')
            # Use MultiLabelBinarizer to create binary columns for multilabel classification
            mlb = MultiLabelBinarizer()
            category = mlb.fit_transform(category)
            category_names = list(mlb.classes_)
                        
            
    return features, category, category_names, data_df

def force_curve_reshape(features):

    max_length = 150
    for i in range (len(features)):
        if(len(features[i][0]<max_length)):
            features[i][0]=np.pad(features[i][0], (0,max_length-len(features[i][0])), 'constant', constant_values=(0))
    feat= np.zeros(len(features) *max_length).reshape(-1, max_length)
    for i in range(len(features)):
        feat[i]= np.concatenate((features[i][0]), axis=None)

    return feat


def acc_curves_reshape(features):

    max_length = 500

    for i in range (len(features)):
        if(len(features[i][0]<max_length)):
            features[i][0]=np.pad(features[i][0], (0,max_length-len(features[i][0])), 'constant', constant_values=(0))
            
        if(len(features[i][1]<max_length)):
            features[i][1]=np.pad(features[i][1], (0,max_length-len(features[i][1])), 'constant', constant_values=(0))
        
        if(len(features[i][2]<max_length)):
            features[i][2]=np.pad(features[i][2], (0,max_length-len(features[i][2])), 'constant', constant_values=(0))
    


    feat= np.zeros(len(features)*3*max_length).reshape(-1, 3*max_length)
    for i in range(len(features)):
        feat[i]= np.concatenate((features[i][0],features[i][1], features[i][2]), axis=None)

    return feat
   