import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import numpy as np

class Plotter:
    def __init__(self):
        plt.ion()  # Mode interactif de Matplotlib
        self.figure, self.ax = plt.subplots()

    def display_image(self, image):
        self.ax.clear()
        #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.ax.imshow(image)
        self.ax.axis('off')  # Ne pas afficher les axes

    def plot_contour(self, contour, color='g', linewidth=2):
        contour_rgb = contour.reshape(-1, 2)
        self.ax.plot(contour_rgb[:, 0], contour_rgb[:, 1], color=color, linewidth=linewidth)

    def scatter_point(self, x, y, color='blue', label=None):
        self.ax.scatter(x, y, color=color, s=40, edgecolor='black', label=label)

    def plot_line(self, x1, y1, x2, y2, color='red', linewidth=2, linestyle='-'):
        self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, linestyle=linestyle)

    def add_text(self, x, y, text, color='white', fontsize=12):
        self.ax.text(x, y, text, color=color, fontsize=fontsize)

    def add_vertical_line(self, x, color='yellow', linestyle='--'):
        self.ax.axvline(x=x, color=color, linestyle=linestyle)

    def show_legend(self):
        self.ax.legend()

    def update_plot(self):
        plt.draw()
        plt.pause(0.001)  # Pause pour l'affichage en temps réel

    def draw_trajectory(self, ref_trajectory, image_shape=(160,120), label='Trajectory', color='blue'):
        """
        Affiche la trajectoire de référence donnée, limitée à la taille de l'image.
        
        Parameters:
        ref_trajectory: np.array
            Trajectoire de référence à afficher, de taille (3, N+1) où N est l'horizon de prédiction.
        image_shape: tuple
            Taille de l'image (hauteur, largeur).
        label: str
            Légende pour la trajectoire.
        color: str
            Couleur de la ligne représentant la trajectoire.
        """
        x_ref = ref_trajectory[0, :]
        y_ref = ref_trajectory[1, :]
        
        # Limiter les coordonnées à la taille de l'image
        height, width = image_shape[:2]
        x_ref = np.clip(x_ref, 0, width - 1)
        y_ref = np.clip(y_ref, 0, height - 1)
        
        self.ax.plot(x_ref, y_ref, color=color, label=label)
        #self.show_legend()
        #self.update_plot()