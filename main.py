import os
import numpy as np
import time
import sys
import torch
import pprint

from train import Trainer
from config import getConfig
cfg = getConfig()


def main():
    print('<---- Training Params ---->')
    pprint.pprint(cfg)

    if cfg.action == 'train':
        runTrain()
    if cfg.action == 'test':
        runTest()

#--------------------------------------------------------------------------------

def runTrain():
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gid

    pathDirData = r'F:\FDAI\NVH\NVH_data/AI_Exterior_Windnoise_image'
    pathDirCSV = r'F:\FDAI\NVH\NVH_data/Inteiror_results_all.csv'

    nnArchitecture = cfg.model
    nnIsTrained = True

    trBatchSize = cfg.b
    teBatchSize = cfg.b
    trMaxEpoch = cfg.epochs

    imgtransResize = cfg.resize
    imgtransCrop = cfg.crop


    pathModel = 'results/{}_{}_{}_{}_{}.pth.tar'.format(cfg.model, cfg.op, cfg.resize, cfg.crop, cfg.gid)
    # check_points = 'results/DENSE-NET-121'
    check_points = None

    print('Training NN architecture = ', nnArchitecture)
    Trainer.train(pathDirData, pathDirCSV, nnArchitecture, nnIsTrained,
                         trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, pathModel, check_points)


    print('Testing the trained model')
    Trainer.test(pathDirData, pathDirCSV, pathModel, nnArchitecture,
                    nnIsTrained, teBatchSize, imgtransResize, imgtransCrop)

#--------------------------------------------------------------------------------

def runTest():
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gid

    pathDirData = r'F:\FDAI\NVH\NVH_data/AI_Exterior_Windnoise_image'
    pathDirCSV = r'F:\FDAI\NVH\NVH_data/Inteiror_results_all.csv'

    nnArchitecture = cfg.model
    nnIsTrained = False
    teBatchSize = cfg.b
    imgtransResize = cfg.resize
    imgtransCrop = cfg.crop

    # pathModel = 'results/{}_{}_{}_{}_{}.pth.tar'.format(cfg.model, cfg.op, cfg.resize, cfg.crop, cfg.gid)
    pathModel = 'results/DENSE-NET-121_adam_256_224_0,1.pth.tar'


    print('Testing the trained model')
    Trainer.test(pathDirData, pathDirCSV, pathModel, nnArchitecture, nnIsTrained, teBatchSize,
             imgtransResize, imgtransCrop)

#--------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
