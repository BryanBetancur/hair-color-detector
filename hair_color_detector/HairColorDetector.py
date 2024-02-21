import cv2 
import numpy as np 
from PIL import Image
from .HairSegmentator import HairSegmentator
from typing import Union
from sklearn.cluster import KMeans

class HairColorDetector:
    def __init__(self) -> None:
        self.hair_segmentator = HairSegmentator()


    def __open_image(self, image_path):
        pil_image = Image.open(image_path)
        if pil_image.format=="PNG":
            pil_image = pil_image.convert("RGB")
        return np.array(pil_image)


    def __process_segment(self, image, save_result):
        parsing = self.hair_segmentator.get_parsing(image)
        parsing = cv2.resize(parsing, 
                              (image.shape[1],image.shape[0]), 
                              interpolation=cv2.INTER_NEAREST
                        )
        hair_mask = self.hair_segmentator.hair(parsing, part=17)  
        hair_segment = self.__get_hair_segment(image, hair_mask)
        if save_result:
            cv2.imwrite("segment.png", hair_segment)
        return hair_segment, hair_mask


    def __get_hair_segment(self, original_image, hair_mask):
        segmento_rgb = cv2.bitwise_and(original_image, original_image, mask=hair_mask)
        contornos, _ = cv2.findContours(hair_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_x, all_y, all_w, all_h = [], [], [], []

        # Itera sobre todos los contornos
        for contorno in contornos:
            # Encuentra el rectángulo delimitador para el contorno actual
            x, y, w, h = cv2.boundingRect(contorno)
            # Guarda las coordenadas del rectángulo delimitador actual en las listas
            all_x.append(x)
            all_y.append(y)
            all_w.append(w)
            all_h.append(h)

        # Encuentra el rectángulo delimitador para todos los contornos combinados
        x = min(all_x)
        y = min(all_y)
        right_boundary = max(x + w for x, w in zip(all_x, all_w))
        bottom_boundary = max(y + h for y, h in zip(all_y, all_h))
        w = right_boundary - x
        h = bottom_boundary - y

        # Recortar el segmento de la imagen RGB
        segmento_rgb_recortado1 = segmento_rgb[y:y+h, x:x+w]
        mask=hair_mask[y:y+h, x:x+w]
        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(segmento_rgb_recortado1, cv2.COLOR_BGR2GRAY)

        # Crear una máscara donde los píxeles negros sean 0 (totalmente transparentes) y los píxeles blancos sean 255 (totalmente opacos)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Dividir la imagen en canales BGR
        b, g, r = cv2.split(segmento_rgb_recortado1)

        # Invertir la máscara
        mask_inv = cv2.bitwise_not(mask)

        # Aplicar la máscara a cada canal de color para hacer transparentes los píxeles negros
        b = cv2.bitwise_and(b, b, mask=mask_inv)
        g = cv2.bitwise_and(g, g, mask=mask_inv)
        r = cv2.bitwise_and(r, r, mask=mask_inv)
        b, g, r = cv2.split(segmento_rgb_recortado1)
        # Combinar los canales para formar la imagen transparente
        return cv2.merge((r, g, b, mask))


    def get_histogram_similarity(self, 
                                 image_inp1: Union[str, np.ndarray], 
                                 image_inp2: Union[str, np.ndarray], 
                                 method=cv2.HISTCMP_BHATTACHARYYA, 
                                 save_result: bool = False
            ):
        assert (isinstance(image_inp1, str) or isinstance(image_inp1, np.ndarray)) and \
            (isinstance(image_inp2, str) or isinstance(image_inp2, np.ndarray)), \
            "image_inp1 and image_inp2 must be instances of str or np.ndarray"
        assert isinstance(save_result, bool), "save_result must be a boolean"
        if type(image_inp1) == str: image1 = self.__open_image(image_inp1)
        else: image1 = image_inp1
        if type(image_inp2) == str: image2 = self.__open_image(image_inp2)
        else: image2 = image_inp2
        hair_segment1, hair_mask1 = self.__process_segment(image1, save_result)
        hair_segment2, hair_mask2 = self.__process_segment(image2, save_result)
        return compare_histograms(hair_segment1, hair_segment2, method)


    def get_color(self,
                  image_inp: Union[str, np.ndarray],
                  n_clusters: int = 3,
                  save_result: bool = False):
        assert (isinstance(image_inp, str) or isinstance(image_inp, np.ndarray)), \
                "image_inp must be instances of str or np.ndarray"        
        assert isinstance(save_result, bool), "save_result must be a boolean"
        if type(image_inp) == str: image = self.__open_image(image_inp)
        else: image = image_inp
        hair_segment, hair_mask = self.__process_segment(image, save_result)
        return hair_segment, hair_mask, dominant_color(hair_segment, n_clusters)


def dominant_color(image, n_clusters=3):
    # Obtener los píxeles no transparentes
    transparent_pixels = np.where(image[:, :, 3] != 0)
    hair_pixels = image[transparent_pixels][:, :3]
    
    # Aplicar KMeans para encontrar los clusters de colores
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(hair_pixels)
    
    # Obtener los centroides de los clusters y sus etiquetas
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Encontrar el color más dominante
    dominant_label = np.argmax(np.bincount(labels))
    dominant_color = centroids[dominant_label]
    
    return dominant_color.astype(int)[::-1]


def compare_histograms(image1, image2, method):
    # Convierte las imágenes a espacio de color RGB
    img1_rgb = image1
    img2_rgb = image2
    
    # Extraer el canal alfa de las imágenes
    alpha_img1 = img1_rgb[:, :, 3]
    alpha_img2 = img2_rgb[:, :, 3]
    
    # Generar una máscara donde los píxeles transparentes tengan un valor de 0
    mask_img1 = np.where(alpha_img1 == 255, 1, 0).astype(np.uint8)
    mask_img2 = np.where(alpha_img2 == 255, 1, 0).astype(np.uint8)
    
    # Aplicar la máscara a las imágenes
    img1_rgb_masked = cv2.bitwise_and(img1_rgb[:, :, :3], img1_rgb[:, :, :3], mask=mask_img1)
    img2_rgb_masked = cv2.bitwise_and(img2_rgb[:, :, :3], img2_rgb[:, :, :3], mask=mask_img2)
    
    # Calcula los histogramas de las imágenes con la máscara aplicada
    hist_img1 = cv2.calcHist([img1_rgb_masked], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_img2 = cv2.calcHist([img2_rgb_masked], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    # Normaliza los histogramas
    cv2.normalize(hist_img1, hist_img1)
    cv2.normalize(hist_img2, hist_img2)
    
    # Calcula la similitud entre los histogramas usando la distancia Chi-Cuadrado
    similarity = cv2.compareHist(hist_img1, hist_img2, method)
    min_similarity = 0  # Valor mínimo deseado
    max_similarity = 1  # Valor máximo deseado
    
    # Normalizar la similitud
    normalized_similarity = (similarity - min_similarity) / (max_similarity - min_similarity)
    
    return normalized_similarity    


if __name__ == "__main__":
    hair_color_detector = HairColorDetector()
    similarity = hair_color_detector.get_histogram_similarity('test1.png','test2.png')
    print('Hair Similarity: ', similarity)