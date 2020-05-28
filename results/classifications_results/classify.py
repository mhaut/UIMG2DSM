import copy
import gc
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from operator import truediv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from sklearn.model_selection import train_test_split
import scipy.io as sio
import sys
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical as keras_to_categorical



def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(y_test, y_pred, numlabels):
    classification = classification_report(y_test, y_pred, labels=range(numlabels))
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)
    return [oa, aa, kappa]


def get_model(namemodel, bands, numclass):
    if namemodel == "RF":
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1, )
    elif namemodel == "MLP":
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(bands,)))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(numclass, activation='softmax'))
        model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
    return model





image = sio.loadmat("generated_maps/OPTICAL.mat")['valor']
data_IMG2 = sio.loadmat("generated_maps/predicted_IMG2DSM.mat")['valor']

listdsms = ["pass", "predicted_CYCLEGAN.mat", "predicted_PROPOSED.mat", "predicted_IMG2DSM.mat", "DSM.mat"]
listnames = ["0optical", "1cycle", "2unit", "3img2dsm", "4original"]

namemodel = sys.argv[1]
export_img = False

for idlist, (acdsm, nameimg) in enumerate(zip(listdsms, listnames)):
    if idlist == 0: continue
    for i in range(10):
        acimage = copy.deepcopy(image)
        if acdsm != "pass":
            valuesdsm = sio.loadmat("generated_maps/" + acdsm)['valor']
            valuesdsm = valuesdsm.reshape(valuesdsm.shape[0], valuesdsm.shape[1], 1)
            data_2use = np.concatenate((acimage, valuesdsm), axis=2)
            del valuesdsm
        else:
            data_2use = acimage
        del acimage

        gt = sio.loadmat("potsdam_test_gt.mat")['gt']
        data_IMG2[np.isnan(data_IMG2)] = 0
        gt[data_IMG2==0] = 0

        colors = [(1, 1, 1), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
        n_bin  = 6
        cm = LinearSegmentedColormap.from_list("a", colors, N=n_bin)

        gt = gt.reshape(-1)

        orshape = data_2use.shape[:-1]
        img2export = data_2use[:1000,1437:2930,:]
        data_2use = data_2use.reshape(-1, data_2use.shape[-1])

        data_2use[np.isnan(data_2use)] = 0
        a = data_2use[gt!=0,:]; b = gt[gt!=0] - 1

        num_classes = len(np.unique(b))

        a /= 255.
        X_train, X_test, y_train, y_test = train_test_split(a, b, test_size=0.95, random_state=69+i, stratify=b)
        del a
        clf = get_model(namemodel, X_train.shape[-1], num_classes)
        
        if namemodel == "RF":
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        elif namemodel == "MLP":
            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.95, random_state=69+i, stratify=y_test)
            clf.fit(X_train, keras_to_categorical(y_train, num_classes),
                            batch_size=4000,
                            epochs=200,
                            verbose=0,
                            validation_data=(X_val, keras_to_categorical(y_val, num_classes)),
                            callbacks = [ModelCheckpoint("best_model.h5", monitor='val_accuracy', verbose=0, save_best_only=True)])
            del clf; K.clear_session(); gc.collect()
            clf = load_model("best_model.h5")
            y_pred = np.argmax(clf.predict(X_test), axis=1)
     
        re = reports(y_test, y_pred, num_classes)
        print(namemodel, nameimg, i, re[0], re[1], re[2])

        if export_img:
            orshape = img2export.shape[:-1]
            if namemodel == "RF":
                predictions = clf.predict(img2export.reshape(-1,img2export.shape[-1])).reshape(orshape)
            elif namemodel == "MLP":
                predictions = np.argmax(clf.predict(data_2use), axis=1).reshape(orshape)
            plt.axis('off')
            plt.imshow(predictions+1, cmap=cm)
            plt.gca().set_axis_off()
            plt.show()
            plt.clf()


