import cv2
import numpy as np
import logging
import os

def process_image_stacked(stacked_image):
    if stacked_image is None or stacked_image.size == 0:
        logging.warning("Empty image received!")
        return None, None

    # Découper le quart inférieur droit de l'image
    h, w = stacked_image.shape[:2]
    cropped_image = stacked_image[h//2:, w:]

    # Affichage de l'image découpée
    cv2.imshow("Cropped Image", cropped_image)
    cv2.waitKey(1000)  # Attendre 1 seconde pour l'affichage

    # Convertir l'image en HSV
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    # Affichage de l'image en HSV
    cv2.imshow("HSV Image", hsv)
    cv2.waitKey(1000)  # Attendre 1 seconde pour l'affichage

    # Définir la plage de couleur jaune en HSV
    lower_orange = np.array([5, 50, 50])
    upper_orange = np.array([50, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_orange, upper_orange)


    #lower_blue = np.array([100, 150, 170])
    #upper_blue = np.array([220, 200, 255])
    #mask_yellow = cv2.inRange(hsv, lower_blue, upper_blue)

    # Affichage du masque jaune
    cv2.imshow("Yellow Mask", mask_yellow)
    cv2.waitKey(1000)  # Attendre 1 seconde pour l'affichage

    # Appliquer le masque sur l'image HSV
    hsv_masked = cv2.bitwise_and(hsv, hsv, mask=mask_yellow)

    # Convertir l'image masquée HSV en BGR pour l'affichage
    masked_image = cv2.cvtColor(hsv_masked, cv2.COLOR_HSV2BGR)

    # Affichage de l'image avec le masque jaune appliqué
    cv2.imshow("Masked Image (Yellow)", masked_image)
    cv2.waitKey(1000)  # Attendre 1 seconde pour l'affichage

    # Trouver les contours dans le masque jaune
    contours_info = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[-2] if len(contours_info) == 3 else contours_info[0]

    # Initialisation des variables pour trouver le contour le plus proche du bas à droite
    closest_contour = None
    closest_distance = float('inf')

    if contours:
        for contour in contours:
            # Calculer le centre du contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Calculer la distance du centre au bas droit de l'image
                distance_to_bottom_right = np.sqrt((w - cx) ** 2 + (h - cy) ** 2)  # Distance au bas droit de l'image

                # Sélectionner le contour le plus proche du bas à droite
                if distance_to_bottom_right < closest_distance:
                    closest_contour = contour
                    closest_distance = distance_to_bottom_right

        if closest_contour is not None:
            # Extraire les points du contour
            points = closest_contour.reshape(-1, 2)

            # Calculer la régression linéaire des points du contour
            [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

            # Calculer les points pour dessiner la ligne
            lefty = int((-x * vy / vx) + y)
            righty = int(((cropped_image.shape[1] - x) * vy / vx) + y)

            # Assurez-vous que les points de la ligne sont des tuples d'entiers
            pt1 = (cropped_image.shape[1] - 1, righty)
            pt2 = (0, lefty)

            # Dessiner la ligne sur l'image originale
            image_with_line = cropped_image.copy()
            cv2.line(image_with_line, pt1, pt2, (0, 255, 0), 2)

            # Affichage de l'image avec la ligne détectée
            cv2.imshow("Image with Line", image_with_line)
            cv2.waitKey(1000)  # Attendre 1 seconde pour l'affichage

            # Retourner l'équation de la droite (vx, vy, x, y) et l'erreur
            center_x = cropped_image.shape[1] // 2
            error = x - center_x
            print('Error:', error)
            return (vx, vy, x, y), error

    logging.warning("No suitable contours found")
    return None, None

# Chemins vers les 4 images à superposer
image_paths = [
    '/Users/ambrericouard/sdsandbox/mycar/data/images/13180_cam_image_array_.jpg',
    '/Users/ambrericouard/sdsandbox/mycar/data/images/13181_cam_image_array_.jpg',
    '/Users/ambrericouard/sdsandbox/mycar/data/images/13182_cam_image_array_.jpg',
    '/Users/ambrericouard/sdsandbox/mycar/data/images/13183_cam_image_array_.jpg'
]

#image_paths = [
#    '/Users/ambrericouard/sdsandbox/mycar/data/images/13240_cam_image_array_.jpg',
#    '/Users/ambrericouard/sdsandbox/mycar/data/images/13241_cam_image_array_.jpg',
#    '/Users/ambrericouard/sdsandbox/mycar/data/images/13242_cam_image_array_.jpg',
#    '/Users/ambrericouard/sdsandbox/mycar/data/images/13243_cam_image_array_.jpg'
#]

image_paths = [
    '/Users/ambrericouard/sdsandbox/mycar/data/images/13270_cam_image_array_.jpg',
    '/Users/ambrericouard/sdsandbox/mycar/data/images/13271_cam_image_array_.jpg',
    '/Users/ambrericouard/sdsandbox/mycar/data/images/13272_cam_image_array_.jpg',
    '/Users/ambrericouard/sdsandbox/mycar/data/images/13273_cam_image_array_.jpg'
]

image_paths = [
    '/Users/ambrericouard/sdsandbox/mycar/data/images/13430_cam_image_array_.jpg',
    '/Users/ambrericouard/sdsandbox/mycar/data/images/13431_cam_image_array_.jpg',
    '/Users/ambrericouard/sdsandbox/mycar/data/images/13432_cam_image_array_.jpg',
    '/Users/ambrericouard/sdsandbox/mycar/data/images/13433_cam_image_array_.jpg'
]

image_paths = [
    '/Users/ambrericouard/sdsandbox/mycar/data/images/13480_cam_image_array_.jpg',
    '/Users/ambrericouard/sdsandbox/mycar/data/images/13481_cam_image_array_.jpg',
    '/Users/ambrericouard/sdsandbox/mycar/data/images/13482_cam_image_array_.jpg',
    '/Users/ambrericouard/sdsandbox/mycar/data/images/13483_cam_image_array_.jpg'
]

image_paths = [
'/Users/ambrericouard/Desktop/Capture d’écran 2024-07-16 à 13.03.11.png',
'/Users/ambrericouard/Desktop/Capture d’écran 2024-07-16 à 13.03.11.png',
'/Users/ambrericouard/Desktop/Capture d’écran 2024-07-16 à 13.03.11.png',
'/Users/ambrericouard/Desktop/Capture d’écran 2024-07-16 à 13.03.11.png'
]

# Initialisation de l'image superposée avec la première image
stacked_image = cv2.imread(image_paths[0])

# Lecture et superposition des 4 images avec des poids progressifs
alpha = 0.5
for i in range(1, len(image_paths)):
    image = cv2.imread(image_paths[i])
    beta = 1.0 - alpha
    stacked_image = cv2.addWeighted(stacked_image, alpha, image, beta, 0)
    alpha += 1.0 / (i + 2)  # Ajustement progressif du poids

stacked_image = np.uint8(stacked_image)  # Conversion en type d'image correct

# Processus de traitement de l'image superposée
line_params, error = process_image_stacked(stacked_image)
if line_params is not None:
    print('Final error:', error)

cv2.waitKey(0)
cv2.destroyAllWindows()