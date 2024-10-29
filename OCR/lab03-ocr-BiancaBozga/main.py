
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
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pytesseract
import subprocess

from skimage.exposure import exposure


def hamming_distance(s1, s2):
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

def hamming_distance_word(w1, w2):
    # Asigurăm că ambele cuvinte au aceeași lungime, completând cu spații dacă este necesar
    max_length = max(len(w1), len(w2))
    w1 = w1.ljust(max_length)
    w2 = w2.ljust(max_length)
    # Calculăm distanța Hamming între cele două cuvinte
    return sum(ch1 != ch2 for ch1, ch2 in zip(w1, w2))
def calculate_cer_wer(expected_text, recognized_text):
    # Convertim lista de cuvinte în șiruri de caractere
    expected_text = ''.join(expected_text)
    recognized_text = ''.join(recognized_text)


    total_characters = len(expected_text)
    total_words = len(expected_text.split())


    substitutions = deletions = insertions = 0


    for expected_char, recognized_char in zip(expected_text, recognized_text):
        if expected_char != recognized_char:
            if expected_char == ' ':
                insertions += 1
            elif recognized_char == ' ':
                deletions += 1
            else:
                substitutions += 1


    cer = (substitutions + deletions + insertions) / total_characters
    wer = (substitutions + deletions + insertions) / total_words

    return cer, wer
def recognize_text_with_tesseract(image_path):
    # Citim imaginea
    image = Image.open(image_path)
    # Recunoaștem textul din imagine folosind Tesseract
    recognized_text = pytesseract.image_to_string(image)
    # Convertim textul recunoscut într-o listă de cuvinte
    recognized_text = recognized_text.split()
    return recognized_text

if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd='C:/Program Files/Tesseract-OCR/tesseract.exe'

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

    '''
    OCR: Read File using the Read API, extract text - remote
    This example will extract text in an image, then print results, line by line.
    This API call can also extract handwriting style text (not shown).
    '''
    print("===== Read File - remote =====")

    read_image_url =open( "text_ai2.jpg",'rb')

    read_response=computervision_client.read_in_stream(
        image=read_image_url,
    mode="Printed",
    raw=True
    )

    # Get the operation location (URL with an ID at the end) from the response
    read_operation_location = read_response.headers["Operation-Location"]
    # Grab the ID from the URL
    operation_id = read_operation_location.split("/")[-1]

    # Call the "GET" API and wait for it to retrieve the results
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)
    result=[]
    # Print the detected text, line by line
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                print(line.text)
                print(line.bounding_box)
                result.append(line.text)
    print()
    '''
    END - Read File - remote
    '''


    groundTruth = ["Succes in rezolvarea", "tEMELOR la", "LABORAtoaree de", "Inteligenta Artificiala!"]
    #la nivel de caracter
    for recognized_text, expected_text in zip(result, groundTruth):
        print(f"Hamming distance between '{recognized_text}' and '{expected_text}': {hamming_distance(recognized_text, expected_text)}")

    # # compute the performance
    # noOfCorrectLines = sum(i == j for i, j in zip(result, groundTruth))
    # print(noOfCorrectLines)

    #la nivel de cuvnat
    for recognized_text, expected_text in zip(result, groundTruth):
        # Afisăm diferența între numărul de cuvinte recunoscute și așteptate

        # Descompunem textul recunoscut și textul așteptat în cuvinte
        recognized_words = recognized_text.split()
        expected_words = expected_text.split()
        num_recognized_words = len(recognized_words)
        num_expected_words = len(expected_words)

        # Calculăm diferența între numărul de cuvinte recunoscute și așteptate
        word_count_difference = num_recognized_words - num_expected_words
        print(f"Word count difference: {word_count_difference}")
        # Calculăm distanța Hamming pentru fiecare pereche de cuvinte
        word_hamming_distances = [hamming_distance_word(recognized_word, expected_word) for
                                  recognized_word, expected_word in zip(recognized_words, expected_words)]
        # Afisăm distanțele pentru fiecare pereche de cuvinte
        print(f"Hamming distances between recognized and expected words: {word_hamming_distances}")
        # # Calculăm și afișăm distanța medie Hamming la nivel de cuvânt
        # avg_word_hamming_distance = sum(word_hamming_distances) / len(word_hamming_distances)
        # print(f"Average Hamming distance per word: {avg_word_hamming_distance}")

    cer, wer = calculate_cer_wer(groundTruth, result)
    print("CER:", cer)
    print("WER:", wer)

    recognized_text = recognize_text_with_tesseract(read_image_url)
    cer1, wer1 = calculate_cer_wer(groundTruth, recognized_text)
    print("CER with Tesseract:", cer1)
    print("WER with Tesseract:", wer1)

    from pytesseract import *
    import argparse
    import cv2

    # We load the input image and then convert
    # it to RGB from BGR. We then use Tesseract
    # to localize each area of text in the input
    # image
    # Citirea imaginiire
    image_path = "text_ai2.jpg"
    images = cv2.imread(image_path)
    rgb = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

    # Extrage textul folosind Tesseract OCR
    results = pytesseract.image_to_data(rgb, output_type=Output.DICT)

    # Parcurge fiecare localizare individuală a textului
    for i in range(0, len(results["text"])):
        # Extrage coordonatele chenarului de delimitare a textului din rezultatul curent
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]

        # Extrage textul OCR și confidența asociată localizării textului
        text = results["text"][i]
        conf = int(results["conf"][i])

        # Filtrează localizările de text cu confidență scăzută
        if conf > 0.25:
            # Afiseaza confidența și textul în terminal
            print("Confidence: {}".format(conf))

            # Adaugă un chenar în jurul textului și afișează textul pe imagine
            cv2.rectangle(images, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(images, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    # Afișează imaginea rezultatului
    plt.imshow(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))
    plt.show()

    # Setarea căii către imagine
    image_path = 'text_ai2.jpg'

    img = cv2.imread(image_path)

    # Conversia imaginii în grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectarea textului folosind Pytesseract
    results = pytesseract.image_to_data(img, output_type=Output.DICT)

    # Afișarea imaginii și desenarea chenarelor pentru text
    for i in range(len(results['text'])):
        # Extrage coordonatele și dimensiunile chenarului
        x = results['left'][i]
        y = results['top'][i]
        w = results['width'][i]
        h = results['height'][i]

        # Extrage textul și scorul de încredere asociat
        text = results['text'][i]
        conf = int(results['conf'][i])

        # Verifică dacă textul este gol și ignoră spațiile goale
        if text.strip():
            # Desenează chenarul și afișează textul doar pentru zonele cu text detectat
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Afișarea imaginii cu chenarele și textul detectat
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    # Load the image
    image_path = "test2.jpeg"

    # load the input image and convert it to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # threshold the image using Otsu's thresholding method
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    recognized_text = pytesseract.image_to_string(thresh)

    recognized_text = recognized_text.split()
    print(recognized_text)
    cer, wer = calculate_cer_wer(groundTruth, recognized_text)
    print("CER:", cer)
    print("WER:", wer)
    results = pytesseract.image_to_data(thresh, output_type=Output.DICT)

    # Afișarea imaginii și desenarea chenarelor pentru text
    for i in range(len(results['text'])):
        # Extrage coordonatele și dimensiunile chenarului
        x = results['left'][i]
        y = results['top'][i]
        w = results['width'][i]
        h = results['height'][i]

        # Extrage textul și scorul de încredere asociat
        text = results['text'][i]
        conf = int(results['conf'][i])

        # Verifică dacă textul este gol și ignoră spațiile goale
        if text.strip():
            # Desenează chenarul și afișează textul doar pentru zonele cu text detectat
            cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(thresh, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Afișarea imaginii cu chenarele și textul detectat
    plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
    plt.show()



