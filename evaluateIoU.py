import os
import numpy as np

from itertools import groupby
from operator import itemgetter

from dataset import load_labels_from_bin
# from config import parser

# args = parser.parse_args()
gt = load_labels_from_bin('thumos14_test_gt.bin', 'test')
predict = {}
with open('thumos14_test_predict_alexnet.bin', 'rb') as f:
    num_videos = np.fromfile(f, dtype=np.int32, count=1)[0]
    num_frames = np.fromfile(f, dtype=np.int32, count=num_videos)
    labels = (np.fromfile(f, dtype=np.float)).reshape(sum(num_frames),21)
    haveread = 0
    test = [293, 324, 664, 714, 767, 864, 873, 1159, 1209, 1255] # 664
    for idx, n in enumerate(num_frames):
        idx = test[idx]
        predict['test_'+str(idx).zfill(7)] = labels[haveread:haveread+n]
        haveread += n

# def IoU(a, b):
#     l = max(len_a+start_a, len_b+start_b) - min(start_a, start_b)
#     temp_a = [-1]*l
#     temp_a[start_a: len_a]
#     return ret

test_fps = [30, 30, 30, 30, 30, 30, 30, 30, 30, 25]
val_fps = [24, 25, 30, 30, 25, 30, 30, 30, 30, 30]

IOU = [0.3, 0.4, 0.5, 0.6, 0.7]
gtclasses = {1:'BaseballPitch', 2:'BasketballDunk', 3:'Billiards', 4:'CleanAndJerk', 5:'CliffDiving',\
    6:'CricketBowling', 7:'CricketShot', 8:'Diving', 9:'FrisbeeCatch', 10:'GolfSwing', 11:'HammerThrow',\
    12:'HighJump', 13:'JavelinThrow', 14:'LongJump', 15:'PoleVault', 16:'Shotput', 17:'SoccerPenalty',\
    18:'TennisSwing', 19:'ThrowDiscus', 20:'VolleyballSpiking', 0:'background'}

def groupable(lefts, right, thres):
    numerator = right[1]-right[0]+1
    denominator = right[1]-lefts[0][0]+1
    for g in lefts:
        numerator += (g[1]-g[0]+1)
    res = numerator/denominator
    return res


for idx, videoid in enumerate(predict.keys()):
    print(videoid)
    if videoid not in gt:
        print(videoid + ' is not available in the ground truth label')
        continue
    p = predict[videoid]
    t = gt[videoid]
    if p.shape[0] != len(t):
        print('numbers of frames in '+ videoid + ' are not matched')
        continue
    pp =  np.argmax(p, axis=1)
    if np.all(np.unique(pp) == np.unique(t)):
        # print(np.unique(pp))
        if len(np.unique(pp)) == 1:
            print(videoid + ' is a background video')
        else:
            print(videoid + ' is predicted correctly as ' + gtclasses[np.unique(pp)[1]])
    else:
        # print(np.unique(pp))
        # print(np.unique(t))
        pass

    locations = {}
    regions = {}
    refined = {}
    for t in np.unique(pp):
        if t==0: # background
            continue
        locations[gtclasses[t]] = np.where(pp==t)[0]
        regions[gtclasses[t]] = []
        refined[gtclasses[t]] = []
    for k,v in locations.items():
        # ref: https://stackoverflow.com/questions/2154249/identify-groups-of-continuous-numbers-in-a-list
        for _, g in groupby(enumerate(list(v)), lambda x: x[0]-x[1]):
            group = list(map(itemgetter(1), g))
            regions[k].append((group[0], group[-1]))

    thres = 0.7
    for clsid,ranges in regions.items():
        lefts = []
        lefts.append(list(ranges[0]))
        for i in range(1, len(ranges)):
            print(lefts)
            print(ranges[i])
            print(groupable(lefts, ranges[i], thres))
            if groupable(lefts, ranges[i], thres)>thres:
                lefts.append(list(ranges[i]))
                prev = groupable(lefts, ranges[i], thres)
            else:
                if lefts[-1][-1] - lefts[0][0] > 20:
                    refined[clsid].append((lefts[0][0],lefts[-1][-1], prev))
                lefts = [list(ranges[i])]
    # print(refined)

    with open('predict2.txt', 'a') as f:
        for k,v in refined.items():
            for x in v:
                f.write('video_{id} {:.03f} {:.03f} {cls} {:.05f}\n'.format(\
                float(x[0]/test_fps[idx]), float(x[1]/test_fps[idx]), \
                float(x[2]), id=videoid, \
                cls=int(list(gtclasses.keys())[list(gtclasses.values()).index(k)]) ))
