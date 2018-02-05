'''
FN_OpenPose.py
Proecess OpenPose .json files and create body part mapping
'''


import json
import multiprocessing as mp
import numpy as np
import os
import random
from scipy.io import (loadmat, savemat)
from skimage import (draw, transform)


class DEFINE():

    # Input Image Size
    SizeInput = (180, 320)
    # Output Image Size
    SizeTarget = (224, 224)

    # Path
    input_path = '/home/khui3/tmp'#'/home/khui3/thumos14_test_keypoints'
    output_path = '/home/khui3/thumos14_pose/val'

    # Multiprocessing
    workers_num = 10

    # Number of Parts of openpose
    openpose_parts_num = 18
    openpose_parts_length = 54

    # Size of parts
    UpArmRatio = 4
    LowArmRatio = 5
    UpLegRatio = 3.5
    LowLegRatio = 4.5
    HandRatio = 4
    FootRatio = 8

    # OpenPose Parts mapping
    mapIdx = {
        # This Part for Mat Notation
        'neck': 1,
        'face': 0,
        'right shoulder': 2,
        'left shoulder': 5,
        'right hip': 8,
        'left hip': 11,
        'right elbow': 3,
        'left elbow': 6,
        'right knee': 9,
        'left knee': 12,
        'right wrist': 4,
        'left wrist': 7,
        'right ankle': 10,
        'left ankle': 13,
        # This Part for Drawing Index
        'head': 1,
        'body': 2,
        'right upper arm': 3,
        'right lower arm': 4,
        'right hand': 5,
        'left upper arm': 6,
        'left lower arm': 7,
        'left hand': 8,
        'right upper leg': 9,
        'right lower leg': 10,
        'right foot': 11,
        'left upper leg': 12,
        'left lower leg': 13,
        'left foot': 14,
    }


# Load json file and parse info into peopleXY
def parse_json(filename):

    # Extract .json into Dict
    with open(filename, 'r') as f:
        data = json.load(f)

    people = data['people']

    # No people detected
    if not people:
        return None

    # Build container
    peopleXY = np.zeros((DEFINE.openpose_parts_num, 3, len(people)))
    # Loop throught the people
    for pidx in range(len(people)):
        key_points = people[pidx]['pose_keypoints']
        assert len(key_points) == DEFINE.openpose_parts_length
        peopleXY[:, :, pidx] = np.reshape(np.array(key_points),
                                          (DEFINE.openpose_parts_num, 3))

    return peopleXY


# Helper Function to draw circle
def imcircle(centerXY, rad):

    rr, cc = draw.circle(centerXY[1], centerXY[0], rad,
                         shape=DEFINE.SizeTarget)
    return rr, cc


# Helper Function to draw Ellipse
def imellipse(oneXY, twoXY, ratio):

    midXY = (oneXY + twoXY) / 2
    c_rad = np.sqrt(np.sum(np.square(oneXY - twoXY))) / 2
    r_rad = c_rad / ratio
    rot = -np.arctan2(oneXY[1] - twoXY[1], oneXY[0] - twoXY[0])
    rr, cc = draw.ellipse(midXY[1], midXY[0], r_rad, c_rad,
                          shape=DEFINE.SizeTarget,
                          rotation=rot)
    return rr, cc


# Draw Head
def imhead(frameXY):

    faceXY = frameXY[:, DEFINE.mapIdx['face']]
    neckXY = frameXY[:, DEFINE.mapIdx['neck']]
    faceXY = (faceXY + neckXY) / 2
    rad = np.sqrt(np.sum(np.square(faceXY - neckXY)))
    rr, cc = imcircle(faceXY, rad)

    return rr, cc


# Draw Body
def imbody(frameXY):

    midXY = (frameXY[:, DEFINE.mapIdx['face']]
             + frameXY[:, DEFINE.mapIdx['neck']]) / 2

    rc = np.array([frameXY[:, DEFINE.mapIdx['left shoulder']],
                   midXY,
                   frameXY[:, DEFINE.mapIdx['right shoulder']],
                   frameXY[:, DEFINE.mapIdx['right hip']],
                   frameXY[:, DEFINE.mapIdx['left hip']]])
    rr, cc = draw.polygon(rc[:, 1], rc[:, 0], shape=DEFINE.SizeTarget)

    return rr, cc

# Draw Upper Right Arm
def imarmUpR(frameXY):

    oneXY = frameXY[:, DEFINE.mapIdx['right shoulder']]
    twoXY = frameXY[:, DEFINE.mapIdx['right elbow']]
    rr, cc = imellipse(oneXY, twoXY, DEFINE.UpArmRatio)
    return rr, cc

# Draw Lower Right Arm
def imarmLowR(frameXY):

    oneXY = frameXY[:, DEFINE.mapIdx['right elbow']]
    twoXY = frameXY[:, DEFINE.mapIdx['right wrist']]
    rr, cc = imellipse(oneXY, twoXY, DEFINE.LowArmRatio)
    return rr, cc

# Draw Upper Left Arm
def imarmUpL(frameXY):

    oneXY = frameXY[:, DEFINE.mapIdx['left shoulder']]
    twoXY = frameXY[:, DEFINE.mapIdx['left elbow']]
    rr, cc = imellipse(oneXY, twoXY, DEFINE.UpArmRatio)
    return rr, cc

# Draw Lower Right Arm
def imarmLowL(frameXY):

    oneXY = frameXY[:, DEFINE.mapIdx['left elbow']]
    twoXY = frameXY[:, DEFINE.mapIdx['left wrist']]
    rr, cc = imellipse(oneXY, twoXY, DEFINE.LowArmRatio)
    return rr, cc

# Draw Upper Right Leg
def imlegUpR(frameXY):

    oneXY = frameXY[:, DEFINE.mapIdx['right hip']]
    twoXY = frameXY[:, DEFINE.mapIdx['right knee']]
    rr, cc = imellipse(oneXY, twoXY, DEFINE.UpLegRatio)
    return rr, cc

# Draw Lower Right Leg
def imlegLowR(frameXY):

    oneXY = frameXY[:, DEFINE.mapIdx['right knee']]
    twoXY = frameXY[:, DEFINE.mapIdx['right ankle']]
    rr, cc = imellipse(oneXY, twoXY, DEFINE.LowLegRatio)
    return rr, cc

# Draw Upper Left Leg
def imlegUpL(frameXY):

    oneXY = frameXY[:, DEFINE.mapIdx['left hip']]
    twoXY = frameXY[:, DEFINE.mapIdx['left knee']]
    rr, cc = imellipse(oneXY, twoXY, DEFINE.UpLegRatio)
    return rr, cc

# Draw Lower Right Arm
def imlegLowL(frameXY):

    oneXY = frameXY[:, DEFINE.mapIdx['left knee']]
    twoXY = frameXY[:, DEFINE.mapIdx['left ankle']]
    rr, cc = imellipse(oneXY, twoXY, DEFINE.LowLegRatio)
    return rr, cc

# Draw Right Hand
def imhandR(frameXY):

    elbowXY = frameXY[:, DEFINE.mapIdx['right elbow']]
    wristXY = frameXY[:, DEFINE.mapIdx['right wrist']]
    dist = np.sqrt(np.sum(np.square(elbowXY - wristXY)))
    rad = dist / DEFINE.HandRatio
    handXY = wristXY + (wristXY - elbowXY) / (dist+1e-10) * rad
    rr, cc = imcircle(handXY, rad)

    return rr, cc

# Draw Left Hand
def imhandL(frameXY):

    elbowXY = frameXY[:, DEFINE.mapIdx['left elbow']]
    wristXY = frameXY[:, DEFINE.mapIdx['left wrist']]
    dist = np.sqrt(np.sum(np.square(elbowXY - wristXY)))
    rad = dist / DEFINE.HandRatio
    handXY = wristXY + (wristXY - elbowXY) / (dist+1e-10) * rad
    rr, cc = imcircle(handXY, rad)

    return rr, cc

# Draw Right Foot
def imfootR(frameXY):

    kneeXY = frameXY[:, DEFINE.mapIdx['right knee']]
    ankleXY = frameXY[:, DEFINE.mapIdx['right ankle']]
    dist = np.sqrt(np.sum(np.square(ankleXY - kneeXY)))
    rad = dist / DEFINE.FootRatio
    footXY = ankleXY + (ankleXY - kneeXY) / (dist+1e-10) * rad
    rr, cc = imcircle(footXY, rad)

    return rr, cc

# Draw Left Foot
def imfootL(frameXY):

    kneeXY = frameXY[:, DEFINE.mapIdx['left knee']]
    ankleXY = frameXY[:, DEFINE.mapIdx['left ankle']]
    dist = np.sqrt(np.sum(np.square(ankleXY - kneeXY)))
    rad = dist / DEFINE.FootRatio
    footXY = ankleXY + (ankleXY - kneeXY) / (dist+1e-10) * rad
    rr, cc = imcircle(footXY, rad)

    return rr, cc

# Work Single json file into figure
def work_single_json(filename):

    peopleXY = parse_json(filename)
    transform_factor = np.flip(np.array(DEFINE.SizeTarget) /
                               np.array(DEFINE.SizeInput),
                               axis=0)

    ret = np.zeros(DEFINE.SizeTarget, dtype=np.uint8)

    # Handle None people
    if peopleXY is None:
        return ret

    # Loop through each people
    for pidx in range(peopleXY.shape[-1]):

        frameXY = (peopleXY[:, :-1, pidx] * transform_factor).T
        confid = peopleXY[:, -1, pidx]

        parts = ['body', 'head',
                 'right upper leg', 'left upper leg',
                 'right lower leg', 'left lower leg',
                 'right foot', 'left foot',
                 'right upper arm', 'left upper arm',
                 'right lower arm', 'left lower arm',
                 'right hand', 'left hand']

        for part in parts:
            if part == 'head':
                if confid[DEFINE.mapIdx['face']] == 0:
                    continue
                if confid[DEFINE.mapIdx['neck']] == 0:
                    continue
                rr, cc = imhead(frameXY)
            elif part == 'body':
                if confid[DEFINE.mapIdx['face']] == 0:
                    continue
                if confid[DEFINE.mapIdx['neck']] == 0:
                    continue
                if confid[DEFINE.mapIdx['right shoulder']] == 0:
                    continue
                if confid[DEFINE.mapIdx['left shoulder']] == 0:
                    continue
                if confid[DEFINE.mapIdx['right hip']] == 0:
                    continue
                if confid[DEFINE.mapIdx['left hip']] == 0:
                    continue
                rr, cc = imbody(frameXY)
            elif part == 'right upper arm':
                if confid[DEFINE.mapIdx['right shoulder']] == 0:
                    continue
                if confid[DEFINE.mapIdx['right elbow']] == 0:
                    continue
                rr, cc = imarmUpR(frameXY)
            elif part == 'right lower arm':
                if confid[DEFINE.mapIdx['right elbow']] == 0:
                    continue
                if confid[DEFINE.mapIdx['right wrist']] == 0:
                    continue
                rr, cc = imarmLowR(frameXY)
            elif part == 'right hand':
                if confid[DEFINE.mapIdx['right elbow']] == 0:
                    continue
                if confid[DEFINE.mapIdx['right wrist']] == 0:
                    continue
                rr, cc = imhandR(frameXY)
            elif part == 'left upper arm':
                if confid[DEFINE.mapIdx['left shoulder']] == 0:
                    continue
                if confid[DEFINE.mapIdx['left elbow']] == 0:
                    continue
                rr, cc = imarmUpL(frameXY)
            elif part == 'left lower arm':
                if confid[DEFINE.mapIdx['left elbow']] == 0:
                    continue
                if confid[DEFINE.mapIdx['left wrist']] == 0:
                    continue
                rr, cc = imarmLowL(frameXY)
            elif part == 'left hand':
                if confid[DEFINE.mapIdx['left elbow']] == 0:
                    continue
                if confid[DEFINE.mapIdx['left wrist']] == 0:
                    continue
                rr, cc = imhandL(frameXY)
            elif part == 'right upper leg':
                if confid[DEFINE.mapIdx['right hip']] == 0:
                    continue
                if confid[DEFINE.mapIdx['right knee']] == 0:
                    continue
                rr, cc = imlegUpR(frameXY)
            elif part == 'right lower leg':
                if confid[DEFINE.mapIdx['right knee']] == 0:
                    continue
                if confid[DEFINE.mapIdx['right ankle']] == 0:
                    continue
                rr, cc = imlegLowR(frameXY)
            elif part == 'right foot':
                if confid[DEFINE.mapIdx['right knee']] == 0:
                    continue
                if confid[DEFINE.mapIdx['right ankle']] == 0:
                    continue
                rr, cc = imfootR(frameXY)
            elif part == 'left upper leg':
                if confid[DEFINE.mapIdx['left hip']] == 0:
                    continue
                if confid[DEFINE.mapIdx['left knee']] == 0:
                    continue
                rr, cc = imlegUpL(frameXY)
            elif part == 'left lower leg':
                if confid[DEFINE.mapIdx['left knee']] == 0:
                    continue
                if confid[DEFINE.mapIdx['left ankle']] == 0:
                    continue
                rr, cc = imlegLowL(frameXY)
            elif part == 'left foot':
                if confid[DEFINE.mapIdx['left knee']] == 0:
                    continue
                if confid[DEFINE.mapIdx['left ankle']] == 0:
                    continue
                rr, cc = imfootL(frameXY)
            else:
                raise Exception('Invalid Part Name!')
            ret[rr, cc] = DEFINE.mapIdx[part]

    return ret


def work_single_dir(folder, fpose, fstat):

    input_folder = '/'.join((DEFINE.input_path, folder))
    all_files = os.listdir(input_folder)
    filenames = [file for file in all_files if file.endswith('.json')]
    filenames = sorted(filenames)

    # output_file = '/'.join((DEFINE.output_path, folder))

    nframes = len(filenames)
    np.array([nframes], dtype=np.int32).tofile(fstat)

    ret = np.zeros(DEFINE.SizeTarget + (nframes,), dtype=np.uint8)

    for nidx in range(nframes):

        ret[:, :, nidx] = work_single_json('/'.join((input_folder,
                                                     filenames[nidx])))

    np.array(ret, dtype=np.uint8).tofile(fpose)
    # savemat(output_file, {'pose': ret})

    print('Processed folder {} with {} files!'.format(folder, len(filenames)))

    return


def work_multi_dir(folders, fpose, fstat):

    for folder in folders:

        work_single_dir(folder, fpose, fstat)

    return


def work_all():

    assert os.path.isdir(DEFINE.input_path)
    if not os.path.isdir(DEFINE.output_path):
        os.mkdir(DEFINE.output_path)

    folders = sorted(os.listdir(DEFINE.input_path), key=lambda f:int(''.join(filter(str.isdigit, f))))
    # folders = os.listdir(DEFINE.input_path)
    print(folders)
    folders = [folder for folder in folders if not os.path.isfile('/'.join((DEFINE.output_path, folder + '.mat')))]

    # Shuffle and hope for the best
    # random.seed()
    # random.shuffle(folders)

    # chunk_size = len(folders) // DEFINE.workers_num + 1 ### multithread is not useful

    print('Start proccessing .json files...')
    print('Input directory: {}'.format(DEFINE.input_path))
    print('Output directory: {}'.format(DEFINE.output_path))
    print('Number of workers: {}'.format(DEFINE.workers_num))

    fpose = open(os.path.join(DEFINE.output_path, 'output.bin'), 'wb')
    fstat = open(os.path.join(DEFINE.output_path, 'stat.bin'), 'wb')
    np.array([len(folders)], dtype=np.int32).tofile(fstat)

    ### multithread is not useful
    # for wid in range(DEFINE.workers_num):
    #
    #     p = mp.Process(target=work_multi_dir,
    #                    args=(folders[wid*chunk_size:(wid+1)*chunk_size],))
    #     p.start()
    work_multi_dir(folders, fpose, fstat)

    fpose.close()
    fstat.close()

    return

def main(target):

    if target == 'workall':
        work_all()

    return


if __name__ == '__main__':

    print('Running FN_OpenPose.py as main...')
    main('workall')
    #os.system('shutdown -s')
