import torch.utils.data as data
import torch
import os
import numpy as np
import scipy.io as sio
from PIL import Image

# videos from 20 classes that are involved in temporal action detection task
interest = {
'validation':[681,682,683,684,685,686,687,688,689,690,901,902,903,904,905,906,
    907,908,909,910,51,52,53,54,55,56,57,58,59,60,151,152,153,154,155,156,157,
    158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,
    177,178,179,180,181,182,183,184,185,186,187,188,189,190,201,202,203,204,205,
    206,207,208,209,210,261,262,263,264,265,266,267,268,269,270,281,282,283,284,
    285,286,287,288,289,290,311,312,313,314,315,316,317,318,319,320,361,362,363,
    364,365,366,367,368,369,370,411,412,413,414,415,416,417,418,419,420,481,482,
    483,484,485,486,487,488,489,490,661,662,663,664,665,666,667,668,669,670,781,
    782,783,784,785,786,787,788,789,790,851,852,853,854,855,856,857,858,859,860,
    931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,
    950,981,982,983,984,985,986,987,988,989,990],
'test':[172,324,421,664,714,727,747,767,873,910,946,964,1038,1182,1324,1447,
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
    1132,1202,1272,1319,1426,1452,1463]}


def load_inputs(labels, frm_path, mode, flag):
    dataset = []
    vidlist = sorted(os.listdir(os.path.join(frm_path)), key=lambda f:int(''.join(filter(str.isdigit, f))))
    # load all the videos from dataset
    for video in vidlist:
        if int(video[-6:]) not in interest[mode]:
            continue
        labellist = labels[video]
        if len(labellist) != len(os.listdir(os.path.join(frm_path, video))):
            print('Caution: frames and labels for '+video+' do not match')
        # load all the frames for each video
        frmlist = sorted(os.listdir(os.path.join(frm_path, video)))
        for idx, frameid in enumerate(frmlist):
            frameid = int(frameid[:-4])-1
        # for frameid in range(len(labellist)):
            l = int(labellist[idx])####frameid
            if flag == 0: # binary labels
                dataset.append( {'video':video, 'frame':frameid+1, 'label':int(l>0)} )
            elif flag == 1: # only action frames
                if l>0:
                    dataset.append( {'video':video, 'frame':frameid+1, 'label':l-1} )
            elif flag == 2: # the entire datset
                dataset.append({'video':video, 'frame':frameid+1, 'label':l})
    return dataset


def load_labels_from_bin(binfile, mode):
    with open(binfile, 'rb') as f:
        num_videos = np.fromfile(f, dtype=np.int32, count=1)[0]
        num_frames = np.fromfile(f, dtype=np.int32, count=num_videos)
        labels = np.fromfile(f, dtype=np.uint8)
    l = {}
    haveread = 0
    val = [417, 687]
#    val = [411, 413, 415, 417, 419, 681, 683, 685, 687, 689] # 685
#    test = [293, 324, 664, 714, 767, 864, 873, 1159, 1209, 1255] # 664
    for idx, n in enumerate(num_frames):
        if mode == 'validation':
            idx = val[idx]
        if mode == 'test':
            idx = test[idx]
        if idx in interest[mode]: ######+1
            l[mode+'_'+str(idx).zfill(7)] = labels[haveread:haveread+n]
        haveread += n
    return l, list(num_frames)


class MyDataset(data.Dataset):
    """
    Arguments:
        mode: validation/test
        frm_path: Path to frame folder
        pose_path: A binary pose file for all videos
        binfile: A binary file path which stores all the labels
        flag: binary/action/entire labels
        transform: PIL transforms
    """

    def __init__(self, mode, frm_path, pose_path, binfile, flag, transform=None):

        self.mode = mode
        self.frm_path = frm_path
        self.pose_path = pose_path
        self.img_ext = '.png'
        self.transform = transform
        self.flag = flag
        if self.pose_path:
            self.fpose = open(self.pose_path, 'rb')
        else:
            self.fpose = None

        self.labels, self.num_frames = load_labels_from_bin(binfile, self.mode)
        self.dataset = load_inputs(self.labels, self.frm_path, self.mode, self.flag)


    def __getitem__(self, index):
#        print ('\tcalling Dataset:__getitem__ @ idx=%d'%index)
        videoid = self.dataset[index]['video']
        frameid = self.dataset[index]['frame']
        label = self.dataset[index]['label']
        # label = self.labels[videoid][frameid-1]

        # PIL image
        img_path = os.path.join(self.frm_path, videoid, str(frameid).zfill(6)+self.img_ext)
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
        	data = self.transform(img) # torch float tensor [3, 224, 224]

        if self.pose_path:
            offset = sum(self.num_frames[:(int(videoid[-6:])-1)])+frameid-1
            self.fpose.seek(224*224*offset)
            part = np.fromfile(self.fpose, dtype=np.uint8, count=224*224)
            part = np.reshape(part, (224, 224))
            part = np.array(np.expand_dims(part, axis=0), dtype=np.float32)/15
            part = torch.from_numpy(part)

            data = torch.cat((data, part),0)

       # if self.pose_path:
       #     parts = sio.loadmat(os.path.join(self.pose_path, videoid+'.mat'))['pose']
       #     part = np.array(np.expand_dims(parts[:,:,frameid-1], axis=0),dtype=np.float32)
       #     part = torch.from_numpy(part)
       #
       #     part = torch.from_numpy(self.dataset[index]['pose'])
       #     part = part/15
       #     data = torch.cat((data, part), 0) # [4, 224, 224]

        return videoid, data, int(label)

    def __len__(self):
 #       print ('\tcalling Dataset:__len__')
        return len(self.dataset)
