import argparse
import os
import random
import shutil
import time

import cv2

import FaceDataCollector as dataCollector
import FaceTesting as tester
import FaceTraining as trainer


class FaceRecognitionSystem:
    DATASET_LOCATION = "CollectedDataset/"
    TrainingDataSetForClassification_LOCATION = "TrainingDataSetForClassification/"

    @staticmethod
    def read_file(file_path):
        try:
            with open(file_path, "r") as file:
                content = file.read()
        except FileNotFoundError:
            return ""
        return content

    @staticmethod
    def delete_all_files(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    @staticmethod
    def create_new_folder(folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            shutil.rmtree(folder_path, ignore_errors=True)
            os.makedirs(folder_path)

    @staticmethod
    def save_image(image_path, file_path):
        img = cv2.imread(image_path)
        if not os.path.exists(file_path[:file_path.rindex("/")]):
            FaceRecognitionSystem.create_new_folder(file_path[:file_path.rindex("/")])
        cv2.imwrite(file_path, img)

    @staticmethod
    def saveImg(data, test_name="test1"):
        for key, imageList in data.items():
            path = FaceRecognitionSystem.TrainingDataSetForClassification_LOCATION + test_name
            for i in range(0, int(len(imageList) * 0.8)):
                FaceRecognitionSystem.save_image(imageList[i], path + f'/train/{key}/img{i + 1}.png')
            for i in range(int(len(imageList) * 0.8), len(imageList) - 1):
                FaceRecognitionSystem.save_image(imageList[i], path + f'/valid/{key}/img{i + 1}.png')
            for i in range(len(imageList) - 1, len(imageList)):
                FaceRecognitionSystem.save_image(imageList[i], path + f'/test/{key}/img{i + 1}.png')

    @staticmethod
    def MakingTrainingDataset(test_name="test1"):
        path = FaceRecognitionSystem.TrainingDataSetForClassification_LOCATION + test_name
        FaceRecognitionSystem.create_new_folder(path)
        FaceRecognitionSystem.create_new_folder(path + "/train")
        FaceRecognitionSystem.create_new_folder(path + "/valid")
        FaceRecognitionSystem.create_new_folder(path + "/test")
        directory = FaceRecognitionSystem.DATASET_LOCATION + test_name
        data = {}
        for folder_name in os.listdir(directory):
            folder_path = os.path.join(directory, folder_name)
            if os.path.isdir(folder_path):
                image_list = []
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        image_path = os.path.join(folder_path, filename)
                        image_list.append(image_path)
                random.shuffle(image_list)
                data[folder_name] = image_list
        FaceRecognitionSystem.saveImg(data, test_name)

    @staticmethod
    def main():
        parser = argparse.ArgumentParser()

        parser.add_argument('--persons', nargs='+', type=str, default=[], help='Names of persons to add')
        parser.add_argument('--train', action='store_true', default=False,
                            help='Train the model')
        parser.add_argument('--find', action='store_true', default=True,
                            help='Find faces')
        parser.add_argument('--test-cases', type=str, default='case1', help='Name of the test case')
        parser.add_argument('--fast', action='store_true', default=False,
                            help='Enable fast mode')
        args = parser.parse_args()
        print(args)
        # args.train = (args.train == 'True' or args.train == 'true')
        # args.find = (args.find == 'True' or args.find == 'true')
        # args.fast = (args.fast == 'True' or args.fast == 'true')
        cam = None
        if len(args.persons) > 0 or args.find:
            print("Installing the camera for capturing Image. It will take a few seconds. Wait ...")
            cam = cv2.VideoCapture(0)
            cam.set(3, 640)
            cam.set(4, 480)

        if args.persons:
            file_path_txt = os.path.join(FaceRecognitionSystem.DATASET_LOCATION, args.test_cases,
                                         f"{args.test_cases}.txt")
            text = FaceRecognitionSystem.read_file(file_path_txt)
            personList = text.split("\n")[:-1]
            for count, personName in enumerate(args.persons):
                file_path_image = os.path.join(FaceRecognitionSystem.DATASET_LOCATION, args.test_cases, personName)
                if personName in personList:
                    print(f"Do you need to re-collect the DataSetForFaceDetection for {personName} again? (Y/n): ",
                          end='')
                    if input().strip().upper() == "Y":
                        FaceRecognitionSystem.delete_all_files(file_path_image)
                print("=" * 50, personName, "=" * 50)
                dataCollector.AddPersons(args.test_cases, personName, cam)
                print("=" * 120)
                if count != len(args.persons) - 1:
                    time.sleep(7)
            if not args.fast:
                cam.release()
                cv2.destroyAllWindows()

        # print('train =>', args.train, type(args.train))
        # print('find =>', args.find, type(args.find))
        # print('fast =>', args.fast, type(args.fast))
        if args.train:
            FaceRecognitionSystem.MakingTrainingDataset(args.test_cases)
            modelPath = os.path.join("runs/classify", args.test_cases)
            if os.path.exists(modelPath):
                shutil.rmtree(modelPath, ignore_errors=True)
            trainer.Train(data_path=os.path.join(FaceRecognitionSystem.TrainingDataSetForClassification_LOCATION,
                                                 args.test_cases), test_name=args.test_cases)
            print("-" * 50, "Find", "-" * 50)
        if args.find:
            modelPath = os.path.join("runs/classify", args.test_cases, "weights", "best.pt")
            tester.FaceRecognition(modelPath, cam, args.fast)
            print("-" * 50, "Find", "-" * 50)


if __name__ == '__main__':
    FaceRecognitionSystem.main()

# python FaceRecognitionSystem.py --find --test-cases case1
# python FaceRecognitionSystem.py --train --find --test-cases case1
# python FaceRecognitionSystem.py --persons ajaySir thippeswamy --train --find --test-cases case1
