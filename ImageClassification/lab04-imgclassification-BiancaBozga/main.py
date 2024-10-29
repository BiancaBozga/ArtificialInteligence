import matplotlib

from math import sqrt
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from array import array
import os
from PIL import Image
import sys
import time

import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pytesseract
import subprocess
import cv2


import torch
from torchvision import transforms
from torchvision.models.detection import ssd300_vgg16
from PIL import Image, ImageDraw


def detect_bikes(image_path,confidence_threshold=0.25):

    image = Image.open(image_path).convert("RGB")

    # Transformă imaginea într-un tensor PyTorch
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)

    # Încarcă modelul SSD pre-antrenat
    model = ssd300_vgg16(pretrained=True)
    model.eval()

    # Detectează obiectele în imagine folosind modelul SSD
    with torch.no_grad():
        predictions = model(image_tensor)

    # Desenează dreptunghiuri separate pentru fiecare bicicletă detectată
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(predictions[0]['scores'], predictions[0]['labels'], predictions[0]['boxes']):
        if label == 2 and score >= confidence_threshold:
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=2)

    # Afisează imaginea cu bicicletele detectate
    image.show()



def euclidean_distance(box1, box2):
    """
    Calculează distanța euclidiană între centrele a două dreptunghiuri.
    """
    # Coordonatele centrului primului dreptunghi
    center1_x = (box1[0] + box1[2]) / 2
    center1_y = (box1[1] + box1[3]) / 2

    # Coordonatele centrului celui de-al doilea dreptunghi
    center2_x = (box2[0] + box2[2]) / 2
    center2_y = (box2[1] + box2[3]) / 2

    # Calculăm distanța euclidiană
    distance = np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)

    return distance
def calculate_iou(box1, box2):

    # Calculăm coordonatele intersecției
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Calculăm suprafața intersecției
    intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

    # Calculăm suprafața totală a ambelor dreptunghiuri
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculăm IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou

def pb1():
    folder_path = 'images'
    real_results=['bike','bike','bike','bike','bike','bike','bike','bike','bike','bike','non-bike','non-bike','non-bike','non-bike','non-bike','non-bike','non-bike','non-bike','non-bike','non-bike']
    computed_results=[]
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        print(image_path)
        # Analyze the image
        with open(image_path, "rb") as image_file:
            result = computervision_client.analyze_image_in_stream(image_file, visual_features=[VisualFeatureTypes.objects])

        # Check if any bicycles are detected in the image
        has_bike = False
        for obj in result.objects:
            if obj.object_property in ["bike", "bicycle"]:
                has_bike = True

                break

        # Add the image to the appropriate array
        if has_bike:
            computed_results.append('bike')
        else:
            computed_results.append('non-bike')
    print(computed_results)

    conf_matrix = confusion_matrix(real_results, computed_results)
    print('Confusion Matrix:')
    print(conf_matrix)
    TP=conf_matrix[1][1]
    TN=conf_matrix[0][0]
    FP=conf_matrix[0][1]
    FN=conf_matrix[1][0]
    #Accuracy
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    print(f"Accuracy:{accuracy}"+'\n')
    print(f"Precision:{precision}"  + '\n')
    print(f"Recall:{recall}" + '\n')

def pb2():

    bike_images_paths=['images/bike01.jpg','images/bike02.jpg','images/bike03.jpg','images/bike04.jpg','images/bike05.jpg',
                       'images/bike06.jpg','images/bike07.jpg','images/bike08.jpg','images/bike09.jpg','images/bike10.jpg']
    bike_images_paths_done_by_me = ['images/done_by_me/bike01.jpg', 'images/done_by_me/bike02.jpg', 'images/done_by_me/bike03.jpg', 'images/done_by_me/bike04.jpg',
                         'images/done_by_me/bike05.jpg','images/done_by_me/bike06.jpg', 'images/done_by_me/bike07.jpg', 'images/done_by_me/bike08.jpg', 'images/done_by_me/bike09.jpg',
                         'images/done_by_me/bike10.jpg']
    done_by_me=[[4.2,26.2,413.1,409.6],[12.4,83.2,383.8,324.1],[60.4,136.5,347.8,410.4],[0.0,0.0,413.8,411.9],
               [65.7,47.2,357.6,348.1],[60.4,135.8,348.3,411.1],[54.4,180,302,413.4],[0,0,389.1,356.4],[0.4,1.5,382.3,411.9],
               [136.2,119.3,378.6,408.1]]
    i=0
    for img_path in bike_images_paths:

        with open(img_path, "rb") as image_file:
             result = computervision_client.analyze_image_in_stream(image_file, visual_features=[VisualFeatureTypes.objects])
        for ob in result.objects:
            if ob.object_property in ["bike", "bicycle"]:
                predicted_bike_bb = [ob.rectangle.x, ob.rectangle.y, ob.rectangle.x + ob.rectangle.w,
                                    ob.rectangle.y + ob.rectangle.h]



        im = plt.imread(img_path)
        fig, ax = plt.subplots()
        ax.imshow(im)

        ax.add_patch(plt.Rectangle(xy=(predicted_bike_bb[0], predicted_bike_bb[1]),
                                         width=predicted_bike_bb[2] - predicted_bike_bb[0],
                                         height=predicted_bike_bb[3] - predicted_bike_bb[1], fill=False, color="green",
                                         linewidth=2))
        left_top = (done_by_me[i][0], done_by_me[i][1])
        right_bottom = (done_by_me[i][2], done_by_me[i][3])

        # Calculăm lungimea și lățimea dreptunghiului
        width = right_bottom[0] - left_top[0]
        height = right_bottom[1] - left_top[1]

        # Creăm un obiect Rectangle și îl adăugăm la axă
        rectangle = patches.Rectangle(left_top, width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rectangle)
        iou = calculate_iou(predicted_bike_bb, done_by_me[i])

        # # Calculăm Average Precision folosind IoU
        # ap = calculate_average_precision(done_by_me[i], [predicted_bike_bb])
        #
        # Calculăm distanța euclidiană între centrele dreptunghiurilor
        de = euclidean_distance(predicted_bike_bb, done_by_me[i])

        ax.text(0.05, 0.85, f'IoU: {iou:.4f}', transform=ax.transAxes, color='white', fontsize=10, ha='left', va='top',
                backgroundcolor='blue')
        # ax.text(0.05, 0.90, f'AP: {ap:.4f}', transform=ax.transAxes, color='white', fontsize=10, ha='left', va='top',
        #         backgroundcolor='blue')
        ax.text(0.05, 0.75, f'DE: {de:.4f}', transform=ax.transAxes, color='white', fontsize=10, ha='left', va='top',
                backgroundcolor='blue')
        plt.show()
        i=i+1

def detectare_ambele_biciclete():
    image_path = "images/bike06.jpg"
    detect_bikes(image_path)

def display_menu():
    print("Meniu:")
    print("1. Pb 1")
    print("2. Pb 2")
    print("3. Detectare ambele bicilete 3")
    print("4. Ieșire")
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

    '''
    Authenticate
    Authenticates your credentials and creates a client.
    '''
    subscription_key = os.environ["VISION_KEY"]
    endpoint = os.environ["VISION_ENDPOINT"]
    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
    '''
    END - Authenticate
    '''
    while True:
        display_menu()
        choice = input("Selectați o opțiune: ")

        if choice == '1':
            print("Ai selectat Opțiunea 1")
            pb1()
        elif choice == '2':
            print("Ai selectat Opțiunea 2")
            pb2()
        elif choice == '3':
            print("Ai selectat Opțiunea 3")
            detectare_ambele_biciclete()
        elif choice == '4':
            print("La revedere!")
            break
        else:
            print("Opțiune invalidă! Te rog să selectezi o opțiune validă.")




