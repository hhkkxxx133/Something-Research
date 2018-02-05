import argparse
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description="PyTorch code to train from scratch")

parser.add_argument('--test_mode', type=str, choices=['test','val'], help='test mode or train mode')
parser.add_argument('--frame_path', metavar='DIR', help='path to dataset frame directories')
parser.add_argument('--gt_path', metavar='DIR', help='path to ground truth file')
parser.add_argument('--pose_path', metavar='DIR', help='path to pose directories')

# ========================= Model Configs ==========================
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: '+' | '.join(model_names)+' (default: resnet18)')
parser.add_argument('--dropout', '--do', default=0.8, type=float,
                    metavar='DO', help='dropout ratio (default: 0.8)')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# parser.add_argument('--training_epoch_multiplier', '--tem', default=10, type=int,
#                     help='replicate the training set by N times in one epoch')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
# parser.add_argument('-i', '--iter-size', default=1, type=int,
#                     metavar='N', help='number of iterations before on update')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
# parser.add_argument('--lr_steps', default=[3, 6], type=float, nargs="+",
#                     metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
#                     metavar='W', help='gradient norm clipping (default: disabled)')
# parser.add_argument('--bn_mode', '--bn', default='frozen', type=str,
#                     help="the mode of bn layers")
# parser.add_argument('--comp_loss_weight', '--lw', default=0.1, type=float,
#                     metavar='LW', help='the weight for the completeness loss')
# parser.add_argument('--reg_loss_weight', '--rw', default=0.1, type=float,
#                     metavar='LW', help='the weight for the location regression loss')

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=5, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')

# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--binary_resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    default=False, help='use pre-trained model')
parser.add_argument('--init_weights', default='', type=str, metavar='PATH',
                    help='path to pretrained weights')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--enable_gpu', dest='enable_gpu', action='store_true',
                    help='option to use cpu/gpu')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
# distributed processing
# parser.add_argument('--world-size', default=1, type=int,
#                     help='number of distributed processes')
# parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                     help='url used to set up distributed training')
# parser.add_argument('--dist-backend', default='gloo', type=str,
#                     help='distributed backend')
