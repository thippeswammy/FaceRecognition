import os

import cv2
from ultralytics import YOLO

from Augmentation import ChangesImage

DATASET_LOCATION = "CollectedDataset/"
NumberOfImageSamples = 500
TotalNumberOfImageSample = 150


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def read_file(file_path):
    try:
        with open(file_path, "r") as file:
            content = file.read()
    except FileNotFoundError:
        return ""
    return content


def AddPersons(TestName: str, personName: str, camera: cv2):
    folder_path = DATASET_LOCATION + TestName
    file_path_image = folder_path + str("/") + personName
    create_folder(folder_path)
    create_folder(file_path_image)
    FaceDataCollection(file_path_image, personName, camera)


def FaceDataCollection(file_path, personName, camera):
    global NumberOfImageSamples, TotalNumberOfImageSample
    count = 0
    FrameRate = 8
    FrameCount = 0
    NumberOfSpacePress = 0
    face_id = personName
    # width, height = 640, 480
    face_detector = YOLO('yolov8m-face.pt')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs like 'XVID' or 'MJPG'
    out = cv2.VideoWriter('output1.avi', fourcc, 20.0, (int(camera.get(3)), int(camera.get(4))))

    # print("press Spacebar for capturing")
    while True:
        ret, img = camera.read()
        # print("======", FrameCount, NumberOfSpacePress, (FrameCount % (30 // FrameRate)), "======")
        FrameCount = FrameCount + 1
        if not ret:
            break
        results = face_detector(img)
        faces = results[0].boxes.xyxy
        if len(faces) > 1:
            cv2.putText(img, 'WARNING: Multiple faces detected!',
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('image', img)
            if cv2.waitKey(1) & 0xff == 27:
                break
            elif count >= NumberOfImageSamples or NumberOfSpacePress >= TotalNumberOfImageSample:
                break
        else:
            for (x1, y1, x2, y2) in faces[0:1]:
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                face_img = img[y1:y2, x1:x2]
                if (cv2.waitKey(1) == 32 or True) and (FrameCount % (30 // FrameRate) == 1):
                    NumberOfSpacePress += 1
                    for i, img_ in enumerate(ChangesImage(face_img)):
                        count += 1
                        cv2.imwrite(f"{file_path}/" + str(face_id) + str(count) + ".jpg", img_)
                # img = cv2.putText(img, f' count:{NumberOfSpacePress}', (width // 2 - 100, height // 2),
                #                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('image', img)
            if cv2.waitKey(1) & 0xff == 27:
                break
            elif count >= NumberOfImageSamples or NumberOfSpacePress >= TotalNumberOfImageSample:
                break

        out.write(img)
    cv2.destroyWindow('image')
