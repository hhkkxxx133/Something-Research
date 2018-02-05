import numpy as np
import os
import scipy.io as sio

from config import parser

args = parser.parse_args()

global gtpath, framepath, gtclasses
gtpath = args.gt_path # "annotation"
framepath = args.frame_path # "/home/khui3/thumos14_frames"
gtclasses = {"BaseballPitch":1, "BasketballDunk":2, "Billiards":3,
	"CleanAndJerk":4, "CliffDiving":5, "CricketBowling":6, "CricketShot":7,
	"Diving":8, "FrisbeeCatch":9, "GolfSwing":10, "HammerThrow":11,
	"HighJump":12, "JavelinThrow":13, "LongJump":14, "PoleVault":15,
	"Shotput":16, "SoccerPenalty":17, "TennisSwing":18, "ThrowDiscus":19,
	"VolleyballSpiking":20, "background":0}

def framepersecond(mode):
	# *_fps.mat is loaded from *_set_meta.mat
	fps_content = sio.loadmat(os.path.join(gtpath, mode+'_fps.mat'))
	fps = fps_content['ans'][0]
	return fps

def annotate(fps, mode):
	annotation = {}
	for gtfile in os.listdir(os.path.join(gtpath,mode)):
		print(gtfile)
		num_suffix = 5+len(mode)
		gtcls = gtfile[:-num_suffix]
		############## TO-DO: ambiguous action label
		if gtcls == 'Ambiguous':
			continue
		gtlabel = gtclasses[gtcls]
		print(gtlabel)
		with open(os.path.join(gtpath,mode,gtfile)) as f:
			for line in f.readlines():
				videoid, start, end = line.split()
				videoid = videoid[6:]
				vid = int(videoid[-7:])
				############### pay attention to the videoid format
				start = int(round(float(start)*fps[vid-1]))
				end = int(round(float(end)*fps[vid-1]))
				if videoid not in annotation:
					annotation[videoid] = [gtlabel,[(start, end)]]
				else:
					if gtlabel not in  annotation[videoid]:
						annotation[videoid].append(gtlabel)
						annotation[videoid].append([(start, end)])
					else:
						idx = annotation[videoid].index(gtlabel)+1
						annotation[videoid][idx].append((start, end))
	return annotation

def perframelabel(annotation, mode):
	raw_data = {} # store per frame labelling for each video
	length = {} # store # of frames for each video
	vidlist = sorted(os.listdir(os.path.join(framepath, mode)), key=lambda f:int(''.join(filter(str.isdigit, f))))
	for video in vidlist:
		############# pay attention to video format
		print(video)
		raw_data[video]=[]
		length[video] = len(os.listdir(os.path.join(framepath, mode, video)))
		if video in annotation:
			for frameid in range(1, length[video]+1):
				flag = False
				if len(annotation[video])==2: # only one label per video
					for pair in annotation[video][1]:
						if frameid in list(range(pair[0],pair[1])):
							raw_data[video].append(annotation[video][0])
							flag = True
							break
				else: # more than one label per video
					# special case of CliffDiving and Diving
					if [5,8] == sorted([annotation[video][0], annotation[video][2]]):
						print(annotation[video]) #########
						idx = annotation[video].index(5)+1
						for pair in annotation[video][idx]:
							if frameid in list(range(pair[0],pair[1])):
								raw_data[video].append(5)
								flag = True
								break
					else:
						for pair in annotation[video][1]:
							if frameid in list(range(pair[0],pair[1])):
								raw_data[video].append(annotation[video][0])
								flag = True
								break
						if flag:
							continue
						for pair in annotation[video][3]:
							if frameid in list(range(pair[0],pair[1])):
								raw_data[video].append(annotation[video][2])
								flag = True
								break
						if len(annotation[video])>4 and not flag:
							for pair in annotation[video][5]:
								if frameid in list(range(pair[0],pair[1])):
									raw_data[video].append(annotation[video][4])
									flag = True
									break
				if not flag:
					raw_data[video].append(0)
		else:
			raw_data[video]=[0]*length[video]
	return length, raw_data

def writebinfile(data_length, raw_data, mode):
	with open(os.path.join(gtpath, 'thumos14_'+mode+'_gt2.bin'),'wb') as f:
		np.array([len(data_length)], dtype=np.int32).tofile(f)
		np.array([v for k,v in sorted(data_length.items())], dtype=np.int32).tofile(f)
		np.array(np.concatenate([v for k,v in sorted(raw_data.items())], axis=0), dtype=np.uint8).tofile(f)

# for now, we use thumos14 validation dataset for training
#                  we use thumos14 test dataset for testing
for mode in ['val', 'test']:
	fps = framepersecond(mode)
	annotation = annotate(fps, mode)
	data_length, raw_data = perframelabel(annotation, mode)
	for v in data_length:
		if data_length[v] != len(raw_data[v]):
			print(v, data_length[v], len(raw_data[v]))
	writebinfile(data_length, raw_data, mode)
