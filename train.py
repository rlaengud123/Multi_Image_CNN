import numpy as np
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from matplotlib import pyplot as plt


from config import getConfig
from DensenetModels import *
from DatasetGenerator import DatasetGenerator


cfg = getConfig()


class Trainer():
    def train(pathDirData, pathDirCSV, nnArchitecture, nnIsTrained,
                         trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, pathModel, checkpoint):

        # -------------------- SETTINGS: NETWORK ARCHITECTURE
        if nnArchitecture == 'DENSE-NET-121':
            model = DenseNet121(nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-161':
            model = DenseNet161(nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169':
            model = DenseNet169(nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201':
            model = DenseNet201(nnIsTrained).cuda()


        model = torch.nn.DataParallel(model).cuda()

        # -------------------- SETTINGS: DATA TRANSFORMS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        transformList = []
        transformList.append(transforms.Resize(imgtransResize))
        transformList.append(transforms.RandomResizedCrop(imgtransCrop))
        transformList.append(transforms.RandomHorizontalFlip())
        # transformList.append(transforms.RandomRotation(5))
        # transformList.append(transforms.ColorJitter(contrast=[0.75, 1.25]))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        transformTrain = transforms.Compose(transformList)

        transformList = []
        transformList.append(transforms.Resize(imgtransResize))
        transformList.append(transforms.RandomResizedCrop(imgtransCrop))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        transformVal = transforms.Compose(transformList)

        # -------------------- SETTINGS: DATASET BUILDERS
        datasetTrain = DatasetGenerator(pathDirData, pathDirCSV, dataset='train', transform=transformTrain)
        datasetVal = DatasetGenerator(pathDirData, pathDirCSV, dataset='val', transform=transformVal)

        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True, num_workers=cfg.workers,
                                     pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=cfg.workers,
                                   pin_memory=True)

        # -------------------- SETTINGS: OPTIMIZER & SCHEDULER
        if cfg.op == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        elif cfg.op == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=0.0001, momentum=0.9)

        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=cfg.patience, mode='min')


        # -------------------- SETTINGS: LOSS
        criterion = nn.MSELoss()

        # ---- Load checkpoint
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'], strict=False)
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
            print('####### Checkpoint Restored #####')

        # ---- TRAIN THE NETWORK
        lossMIN = 100000


        for epochID in range(0, trMaxEpoch):

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime
            print('current lr : {:.7f}'.format(optimizer.param_groups[0]['lr']))


            Trainer.epochTrain(model, dataLoaderTrain, optimizer, criterion)

            lossVal, losstensor = Trainer.epochVal(model, dataLoaderVal, criterion)

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime

            scheduler.step(losstensor.data)

            if lossVal < lossMIN:
                lossMIN = lossVal
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
                            'optimizer': optimizer.state_dict()}, pathModel)

                print('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
            else:
                print('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))

    # --------------------------------------------------------------------------------


    def epochTrain(model, dataLoader, optimizer, criterion):
        model.train()


        for batchID, data in enumerate(tqdm(dataLoader)):
            inputs1, inputs2, inputs3, inputs4, inputs5, y = Variable(data['image1'].cuda()), \
                                                             Variable(data['image2'].cuda()), \
                                                             Variable(data['image3'].cuda()), \
                                                             Variable(data['image4'].cuda()), \
                                                             Variable(data['image5'].cuda()), \
                                                             Variable(data['y'].cuda())

            output1 = model(inputs1)
            output1 = torch.reshape(output1, (-1,))
            loss1 = criterion(output1.float(), y.float())

            output2 = model(inputs2)
            output2 = torch.reshape(output2, (-1,))
            loss2 = criterion(output2.float(), y.float())

            output3 = model(inputs3)
            output3 = torch.reshape(output3, (-1,))
            loss3 = criterion(output3.float(), y.float())

            output4 = model(inputs4)
            output4 = torch.reshape(output4, (-1,))
            loss4 = criterion(output4.float(), y.float())

            output5 = model(inputs5)
            output5 = torch.reshape(output5, (-1,))
            loss5 = criterion(output5.float(), y.float())

            total_loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5


            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    # --------------------------------------------------------------------------------

    def epochVal(model, dataLoader, criterion):

        with torch.no_grad():
            model.eval()

            lossVal = 0
            lossValNorm = 0
            losstensorMean = 0

            for i, data in enumerate(tqdm(dataLoader)):
                inputs1, inputs2, inputs3, inputs4, inputs5, y = Variable(data['image1'].cuda()), \
                                                                 Variable(data['image2'].cuda()), \
                                                                 Variable(data['image3'].cuda()), \
                                                                 Variable(data['image4'].cuda()), \
                                                                 Variable(data['image5'].cuda()), \
                                                                 Variable(data['y'].cuda())

                output1 = model(inputs1)
                output1 = torch.reshape(output1, (-1,))
                loss1 = criterion(output1.float(), y.float())

                output2 = model(inputs2)
                output2 = torch.reshape(output2, (-1,))
                loss2 = criterion(output2.float(), y.float())

                output3 = model(inputs3)
                output3 = torch.reshape(output3, (-1,))
                loss3 = criterion(output3.float(), y.float())

                output4 = model(inputs4)
                output4 = torch.reshape(output4, (-1,))
                loss4 = criterion(output4.float(), y.float())

                output5 = model(inputs5)
                output5 = torch.reshape(output5, (-1,))
                loss5 = criterion(output5.float(), y.float())

                total_loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5

                losstensorMean += total_loss

                lossVal += total_loss
                lossValNorm += 1

            outLoss = lossVal / lossValNorm
            losstensorMean = losstensorMean / lossValNorm

            return outLoss, losstensorMean



    # --------------------------------------------------------------------------------

    def MAPE(y_true, y_pred):
        y_true, y_pred = y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
        return np.mean(np.abs((y_true - y_pred) / y_true)) *100

    # --------------------------------------------------------------------------------

    def load_pretrained(model, pathModel):
        modelCheckpoint = torch.load(pathModel)
        state_dict = modelCheckpoint['state_dict']
        model.load_state_dict(state_dict, strict=False)
        print('###### pre-trained Model restored #####')

    # --------------------------------------------------------------------------------

    def test(pathDirData, pathDirCSV, pathModel, nnArchitecture, nnIsTrained, teBatchSize,
             imgtransResize, imgtransCrop):


        cudnn.benchmark = True

        # -------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        if nnArchitecture == 'DENSE-NET-121':
            model = DenseNet121(nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-161':
            model = DenseNet161(nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169':
            model = DenseNet169(nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201':
            model = DenseNet201(nnIsTrained).cuda()


        model = torch.nn.DataParallel(model).cuda()
        Trainer.load_pretrained(model, pathModel)

        # -------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # -------------------- SETTINGS: DATASET BUILDERS
        transformList = []
        transformList.append(transforms.Resize(imgtransResize))
        transformList.append(transforms.RandomResizedCrop(imgtransCrop))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        transformTest = transforms.Compose(transformList)

        datasetTest = DatasetGenerator(pathDirData, pathDirCSV, dataset='test', transform=transformTest)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=teBatchSize, num_workers=cfg.workers,
                                    shuffle=False, pin_memory=True)


        with torch.no_grad():
            model.eval()
            pred = torch.FloatTensor().cuda()
            label = torch.FloatTensor().cuda()

            for i, data in enumerate(tqdm(dataLoaderTest)):
                inputs1, inputs2, inputs3, inputs4, inputs5, y = Variable(data['image1'].cuda()), \
                                                                 Variable(data['image2'].cuda()), \
                                                                 Variable(data['image3'].cuda()), \
                                                                 Variable(data['image4'].cuda()), \
                                                                 Variable(data['image5'].cuda()), \
                                                                 Variable(data['y'].cuda())
                output1 = model(inputs1)
                output1 = torch.reshape(output1, (-1,))

                output2 = model(inputs2)
                output2 = torch.reshape(output2, (-1,))

                output3 = model(inputs3)
                output3 = torch.reshape(output3, (-1,))

                output4 = model(inputs4)
                output4 = torch.reshape(output4, (-1,))

                output5 = model(inputs5)
                output5 = torch.reshape(output5, (-1,))

                output = (output1 + output2 + output3 + output4 + output5) / 5
                pred = torch.cat((pred, output), 0)
                label = torch.cat((label, y), 0)


            MAPE = Trainer.MAPE(label, pred)
            print(f'Final MAPE : {100-MAPE: .2f}%')


            plt.plot(label.cpu().detach().numpy())
            plt.plot(pred.cpu().detach().numpy())
            plt.legend(["true", "pred"])
            plt.show()

            return
# --------------------------------------------------------------------------------