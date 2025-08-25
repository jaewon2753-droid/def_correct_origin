import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
import glob
from shutil import copyfile
import matplotlib.pyplot as plt
from utilities.customUtils import *
# from dataTools.sampler import * # <--- 1. 이 줄을 삭제했습니다.
import numpy as np
import cv2
from PIL import Image
from dataTools.dataNormalization import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class AddGaussianNoise(object):
    def __init__(self, noiseLevel):
        self.var = 0.1
        self.mean = 0.0
        self.noiseLevel = noiseLevel

    def __call__(self, tensor):
        # noiseLevel이 0이면 노이즈를 추가하지 않음
        if self.noiseLevel == 0:
            return tensor
        sigma = self.noiseLevel/100.
        noisyTensor = tensor + torch.randn(tensor.size()).uniform_(0, 1.) * sigma  + self.mean
        return noisyTensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.var)


class inference():
    def __init__(self, gridSize, inputRootDir, outputRootDir, modelName, resize = None, validation = None ):
        self.inputRootDir = inputRootDir
        self.outputRootDir = outputRootDir
        self.modelName = modelName
        self.resize = resize
        self.validation = validation
        self.unNormalize = UnNormalize()

    def inputForInference(self, imagePath, noiseLevel):
        img = Image.open(imagePath).convert("RGB")

        # ========================================================== #
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 핵심 수정 사항 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
        # ---------------------------------------------------------- #
        # 2. 기존의 강제 리사이즈 및 쿼드 베이어 샘플링 로직을 모두 제거했습니다.
        # 이제 입력 이미지를 그대로 사용합니다.
        # ========================================================== #

        img = np.asarray(img)
        img = Image.fromarray(img)

        # 이미지를 텐서로 변환하고, 정규화하고, 필요한 경우 테스트 노이즈를 추가합니다.
        transform = transforms.Compose([ transforms.ToTensor(),
                                        transforms.Normalize(normMean, normStd),
                                        AddGaussianNoise(noiseLevel=noiseLevel)])

        testImg = transform(img).unsqueeze(0)

        return testImg


    def saveModelOutput(self, modelOutput, inputImagePath, noiseLevel, step = None, ext = ".png"):
        datasetName = os.path.basename(os.path.dirname(inputImagePath))
        if step:
            imageSavingPath = os.path.join(self.outputRootDir, self.modelName, datasetName, f"{extractFileName(inputImagePath, True)}_sigma_{noiseLevel}_{self.modelName}_{step}{ext}")
        else:
            imageSavingPath = os.path.join(self.outputRootDir, self.modelName, datasetName, f"{extractFileName(inputImagePath, True)}_sigma_{noiseLevel}_{self.modelName}{ext}")

        # 저장 경로가 없으면 생성
        os.makedirs(os.path.dirname(imageSavingPath), exist_ok=True)
        save_image(self.unNormalize(modelOutput[0]), imageSavingPath)

    def testingSetProcessor(self):
        # 3. 경로 처리를 os.path.join으로 수정하여 OS 호환성을 높였습니다.
        testSets = glob.glob(os.path.join(self.inputRootDir, '*/'))
        if not testSets: # 하위 폴더가 없으면 현재 폴더에서 이미지 검색
             testSets = [self.inputRootDir]

        if self.validation:
            testSets = testSets[:1]

        testImageList = []
        for t in testSets:
            testSetName = os.path.basename(os.path.normpath(t))
            createDir(os.path.join(self.outputRootDir, self.modelName, testSetName))
            imgInTargetDir = imageList(t, False)
            testImageList.extend(imgInTargetDir)

        return testImageList
