import os
import cv2
import sys
import time
import shutil
import random
from FaceRecognitionByTraining import FaceTesting as tester, FaceTraining as trainer, FaceDataCollector as dataCollector

DATASET_LOCATION = "CollectedDataset/"
TrainingDataSetForClassification_LOCATION = "TrainingDataSetForClassification/"


def read_file(file_path):
    try:
        with open(file_path, "r") as file:
            content = file.read()
    except FileNotFoundError:
        return ""
    return content


def delete_all_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def create_new_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        shutil.rmtree(folder_path, ignore_errors=True)
        os.makedirs(folder_path)


def save_image(image_path, file_path):
    img = cv2.imread(image_path)
    if not os.path.exists(file_path[:file_path.rindex("/")]):
        create_new_folder(file_path[:file_path.rindex("/")])
    cv2.imwrite(file_path, img)


def saveImg(data, test_name="test1"):
    for key, imageList in data.items():
        path = TrainingDataSetForClassification_LOCATION + test_name
        for i in range(0, int(len(imageList) * 0.8)):
            save_image(imageList[i], path + f'/train/{key}/' + str(f"img{i + 1}.png"), )
        for i in range(int(len(imageList) * 0.8), len(imageList) - 1):
            save_image(imageList[i], path + f'/valid/{key}/' + str(f"img{i + 1}.png"))
        for i in range(len(imageList) - 1, len(imageList)):
            save_image(imageList[i], path + f'/test/{key}/' + str(f"img{i + 1}.png"))


def MakingTrainingDataset(test_name="test1"):
    path = TrainingDataSetForClassification_LOCATION + test_name
    create_new_folder(path)
    create_new_folder(path + "/valid")
    create_new_folder(path + "/valid")
    create_new_folder(path + "/test")
    directory = DATASET_LOCATION + test_name
    data = {}
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path):  # Check if it's a directory
            image_list = []
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(
                        (".jpg", ".jpeg", ".png")):  # Check for image extensions (lower case for case-insensitivity)
                    image_path = os.path.join(folder_path, filename)
                    image_list.append(image_path)
            random.shuffle(image_list)
            data[folder_name] = image_list
    saveImg(data, test_name)


if __name__ == "__main__":
    FullCmd = [[], False, False, "test1", None, False]
    cmdLine = sys.argv
    print(cmdLine)
    if len(cmdLine) == 1:
        cmdLine = input("enter => ")
    cmdLine = cmdLine.split(" ")
    for cmd in cmdLine:
        if cmd.split("=")[0] == "Persons":
            FullCmd[0] = cmd.split("=")[1].lower().split(",")
        elif cmd.lower() == "train":
            FullCmd[1] = True
        elif cmd.lower() == "find":
            FullCmd[2] = True
        elif cmd.split("=")[0] == "TestCases":
            FullCmd[3] = cmd.split("=")[1]
        elif cmd.split("=")[0].lower() == "fast":
            FullCmd[4] = True
    cam = None
    if len(FullCmd[0]) > 0 or FullCmd[2]:
        print("Installing the camera for capturing Image. It will take a few seconds. Wait ...")
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)
        cam.set(4, 480)
    else:
        FullCmd[4] = False
    print(FullCmd)
    if len(FullCmd[0]) > 0:
        file_path_txt = DATASET_LOCATION + FullCmd[3] + str("/") + FullCmd[3] + str(".txt")
        if len(FullCmd) == 0:
            print("did not enter the name")
        text = read_file(file_path_txt)
        personList = text.split("\n")[:-1]
        personNameList = FullCmd[0]
        for count, personName in enumerate(personNameList):
            file_path_image = DATASET_LOCATION + FullCmd[3] + str("/") + personName
            if personName in personList:
                print("Do u need  to Re-collect the DataSetForFaceDetection", personName, " again the press (Y/n):",
                      end='')
                if input().upper() == "Y":
                    delete_all_files(file_path_image)
            print("=" * 50, personName, "=" * 50)
            dataCollector.AddPersons(FullCmd[3], personName, cam)
            print("=" * 120)
            if count != len(personNameList) - 1:
                time.sleep(2)
        if not FullCmd[4]:
            cam.release()
            cv2.destroyAllWindows()
        MakingTrainingDataset(FullCmd[3])

    if FullCmd[1]:
        modelPath = "runs/classify/" + FullCmd[3]
        if os.path.exists(modelPath):
            shutil.rmtree(modelPath, ignore_errors=True)
        trainer.Train(data_path=TrainingDataSetForClassification_LOCATION + FullCmd[3], test_name=FullCmd[3])
    if FullCmd[2]:
        modelPath = "runs/classify/" + FullCmd[3] + "/weights/best.pt"
        tester.FaceRecognition(modelPath, cam, FullCmd[4])

# Persons=name1,name2 train find TestCase=fileName

# find TestCases=case1
# train find TestCases=case1
# Persons=ajaySir,thippeswamy train find TestCases=case1
# Persons=thippeswamy,pranay,nithish train find TestCases=case1
