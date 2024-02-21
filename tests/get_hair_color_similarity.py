import cv2 
import numpy as np 
from PIL import Image
from ..HairColorDetector import HairColorDetector

if __name__ == "__main__":
    hair_color_detector = HairColorDetector()
    similarity = hair_color_detector.get_histogram_similarity('test1.png','test2.png')
    print('Hair Similarity: ', similarity)