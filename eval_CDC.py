import os
import scipy.io as sio
import numpy as np
from dataset import load_labels_from_bin
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

CDC = [4, 6, 7, 11, 26, 28, 39, 45, 46, 51, 58, 62, 73, 85, 113, 129, 131, 173,
    179, 188, 211, 220, 238, 242, 250, 254, 270, 273, 278, 285, 292, 293, 308,
    319, 324, 353, 355, 357, 367, 372, 374, 379, 392, 405, 412, 413, 423, 426,
    429, 437, 442, 443, 444, 448, 450, 461, 464, 504, 505, 538, 541, 549, 556,
    558, 560, 569, 577, 591, 593, 601, 602, 611, 615, 617, 622, 624, 626, 635,
    664, 665, 671, 672, 673, 689, 691, 698, 701, 714, 716, 718, 723, 724, 730,
    737, 740, 756, 762, 765, 767, 771, 785, 786, 793, 796, 798, 807, 814, 839,
    844, 846, 847, 854, 864, 873, 882, 887, 896, 897, 903, 940, 946, 950, 964,
    981, 987, 989, 991, 1008, 1038, 1039, 1040, 1058, 1064, 1066, 1072, 1075,
    1076, 1078, 1079, 1080, 1081, 1098, 1114, 1118, 1123, 1127, 1129, 1134,
    1135, 1146, 1153, 1159, 1162, 1163, 1164, 1168, 1174, 1182, 1194, 1195,
    1201, 1202, 1207, 1209, 1219, 1223, 1229, 1235, 1247, 1255, 1257, 1267,
    1268, 1270, 1276, 1281, 1292, 1307, 1309, 1313, 1314, 1319, 1324, 1325,
    1339, 1343, 1358, 1369, 1389, 1391, 1409, 1431, 1433, 1446, 1447, 1452,
    1459, 1460, 1463, 1468, 1483, 1484, 1495, 1496, 1508, 1512, 1522, 1527,
    1531, 1532, 1549, 1556, 1558]

gtclasses = {0:'background', 1:'BaseballPitch', 2:'BasketballDunk',
    3:'Billiards', 4:'CleanAndJerk', 5:'CliffDiving', 6:'CricketBowling',
    7:'CricketShot', 8:'Diving', 9:'FrisbeeCatch', 10:'GolfSwing',
    11:'HammerThrow', 12:'HighJump', 13:'JavelinThrow', 14:'LongJump',
    15:'PoleVault', 16:'Shotput', 17:'SoccerPenalty', 18:'TennisSwing',
    19:'ThrowDiscus', 20:'VolleyballSpiking'}

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        num = cm.sum(axis=1)[:, np.newaxis]
        num[num==0] = 1
        cm = cm.astype('float') / num#cm.sum(axis=1)[:, np.newaxis]
        # print(cm.sum(axis=1)[:, np.newaxis].shape)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    clsacc = 0
    for i in range(len(cm)):
        clsacc += cm[i][i]
    print(len(cm))
    print(clsacc/len(cm))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('ConfusionMatrix.png')

matfile = sio.loadmat('proball.mat')
frameid = matfile['videoid']
print(frameid.shape) # (1154579, 1)
predict = np.transpose(matfile['proball'])
print(predict.shape) # (1154579, 21)

num_frames = []
for v in CDC:
    num_frames.append( len(frameid) - np.count_nonzero(frameid-v) )

gt = load_labels_from_bin('thumos14_test_25fps_gt.bin', 'test')
print(len(gt))

haveread = 0
for idx, v in enumerate(CDC):
    videoid = 'test_'+str(v).zfill(7)
    print('>>> '+videoid)
    if videoid not in gt:
        print(videoid + ' is not available in the ground truth label')
        haveread += num_frames[idx]
        continue

    p = predict[haveread:haveread+num_frames[idx], :]
    haveread += num_frames[idx]

    pp =  np.argmax(p, axis=1)
    t = gt[videoid]
    # print(p.shape[0])
    # print(len(t))
    if p.shape[0] != len(t):
        print('number of frames in predict is: '+str(p.shape[0]))
        print('number of frames in ground truth is: '+str(len(t)))
        # print('numbers of frames in '+ videoid + ' are not matched')
        # continue
        pp = pp[1:]

    if idx == 0:
        y_true = t
        y_pred = pp
    else:
        y_true = np.hstack((y_true, t))
        y_pred = np.hstack((y_pred, pp))

    continue


conf_mat = confusion_matrix(y_true, y_pred)
plt.figure()
plot_confusion_matrix(conf_mat, classes=gtclasses.values(), normalize=True,
                      title='Normalized Confusion Matrix')
plt.show()
