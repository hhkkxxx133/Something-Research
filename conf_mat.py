import numpy as np
import itertools
import matplotlib.pyplot as plt
from dataset import load_labels_from_bin
from sklearn.metrics import confusion_matrix

interest = [172,324,421,664,714,727,747,767,873,910,946,964,1038,1182,1324,1447,
    1486,1544,179,250,331,368,560,601,716,718,821,903,1064,1113,1205,1257,1333,
    1343,1362,51,81,352,412,416,424,544,556,558,577,771,965,1075,1146,1240,1247,
    1267,58,504,505,617,635,698,737,740,1270,1307,1483,131,173,278,285,357,437,
    673,724,730,785,844,920,940,1076,1134,1166,1339,1558,167,267,273,353,392,
    422,502,549,569,646,786,796,897,981,1313,1345,1573,4,237,405,448,568,622,
    650,798,1061,1194,1276,1325,1358,1460,85,254,458,461,602,624,636,715,764,
    807,989,1081,1309,1354,1532,1572,95,174,377,413,588,672,1028,1135,1387,1433,
    1496,1508,1512,1549,28,46,113,238,329,611,625,824,827,847,1043,1127,1163,
    1177,1292,1363,371,372,443,608,654,842,882,950,1058,1072,1137,1164,1184,
    1229,1282,1431,36,124,188,220,308,435,442,479,538,769,839,840,887,896,1008,
    1098,1231,7,293,835,864,886,924,1144,1159,1195,1209,1255,1314,1391,1446,62,
    242,355,379,795,863,1079,1155,1158,1170,1219,1281,1369,1459,1495,1555,32,
    144,178,321,444,464,626,665,667,793,1123,1162,1201,1207,1235,1352,1405,11,
    45,129,367,593,691,708,749,765,846,854,979,1039,1040,1066,1114,1129,1268,
    1451,1475,1527,292,541,591,689,701,991,1118,1153,1409,1484,1556,26,39,374,
    423,518,688,756,1168,1389,1522,1531,73,158,319,671,685,776,814,987,1080,
    1104,1174,1223,1305,1468,1547,6,211,426,429,450,524,615,723,762,1078,1093,
    1132,1202,1272,1319,1426,1452,1463]
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

interest.sort()
test = [293, 324, 664, 714, 767, 864, 873, 1159, 1209, 1255]
# gtclasses = {0:'background', 1:'BaseballPitch', 2:'BasketballDunk',
#     3:'Billiards', 4:'CleanAndJerk', 5:'CliffDiving',6:'CricketBowling',
#     7:'CricketShot', 8:'Diving', 9:'FrisbeeCatch', 10:'GolfSwing',
#     11:'HammerThrow', 12:'HighJump', 13:'JavelinThrow', 14:'LongJump',
#     15:'PoleVault', 16:'Shotput', 17:'SoccerPenalty', 18:'TennisSwing',
#     19:'ThrowDiscus', 20:'VolleyballSpiking'}
# gtclasses = {0:'background', 1:'BaseballPitch', 13:'JavelinThrow'}
gtclasses = {0:'background', 1:'action'}

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        # cm[8][8] += cm[5][5]
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    clsacc = 0
    for i in range(len(cm)):
        clsacc += cm[i][i]
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


gt, _ = load_labels_from_bin('thumos14_test_gt2.bin', 'test')
'''
tot = 0
count = 0
for idx, videoid in enumerate(gt.keys()):
    if int(videoid[-6:]) in CDC:
        print(videoid)
        tot += len(gt[videoid])
        count += 1
print('total number of frames')
print(tot)
print(count)
print("not in")
for t in CDC:
    if t not in interest:
        print(t)
'''
predict={}
with open('thumos14_test_binary_predict_vgg16.bin', 'rb') as f:
    num_videos = np.fromfile(f, dtype=np.int32, count=1)[0]
    num_frames = np.fromfile(f, dtype=np.int32, count=num_videos)
    labels = (np.fromfile(f, dtype=np.float))#######.reshape(sum(num_frames),21)
    haveread = 0
    for idx, n in enumerate(num_frames):
        idx = interest[idx]
        predict['test_'+str(idx).zfill(7)] = labels[haveread:haveread+n]
        haveread += n

# construct confusion matrix
# conf_mat = np.zeros((21, 21))
for idx, videoid in enumerate(predict.keys()):
    print('>>> '+videoid)
    if videoid not in gt:
        print(videoid + ' is not available in the ground truth label')
        continue
    p = predict[videoid]
    t = gt[videoid]
    # print(p.shape[0])
    # print(len(t))
    if p.shape[0] != len(t):
        print('numbers of frames in '+ videoid + ' are not matched')
        continue
    pp = [int(i>0) for i in p]
    t = [int(i>0) for i in t]
    ########### pp =  np.argmax(p, axis=1)

    if idx == 0:
        y_true = t
        y_pred = pp
    else:
        y_true = np.hstack((y_true, t))
        y_pred = np.hstack((y_pred, pp))

    continue
    # if np.all(np.unique(pp) == np.unique(t)):
    #     # print(np.unique(pp))
    #     if len(np.unique(pp)) == 1:
    #         print(videoid + ' is a background video')
    #     else:
    #         print(videoid + ' is predicted correctly as ' + gtclasses[np.unique(pp)[1]])
    # else:
    #     # print(np.unique(pp))
    #     # print(np.unique(t))
    #     pass

conf_mat = confusion_matrix(y_true, y_pred)
plt.figure()
plot_confusion_matrix(conf_mat, classes=gtclasses.values(), normalize=True,
                      title='Normalized Confusion Matrix')
#plt.show()
