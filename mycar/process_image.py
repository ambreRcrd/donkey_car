import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging
import math
#from mpc import MPCController
from plotter import Plotter

class ImageProcessor:
    def __init__(self, image, plotter, mpc):
        self.image = image
        self.plotter = plotter
        self.steering_values = []
        self.throttle_values = []
        self.errors = []
        self.Y_values = []
        self.theta_values = [0]
        self.theta = 0
        self.xd = 0
        self.yd = 0
        self.xd_prev = 80
        self.yd_prev = 60
        self.xd_prev_prev = 80
        self.yd_prev_prev = 60
        self.plot = True
        self.distance_beteween_contours = 95
        self.distance = 60
        self.mpc = mpc

    def preprocess_image(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)  # RGB -> BGR
        self.image_height, self.image_width, _ = self.image.shape

    def apply_mask(self, lower_h, lower_s, lower_v, upper_h, upper_s, upper_v):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        lower_color = np.array([lower_h, lower_s, lower_v])
        upper_color = np.array([upper_h, upper_s, upper_v])
        mask = cv2.inRange(hsv, lower_color, upper_color)
        masked_image = cv2.bitwise_and(self.image, self.image, mask=mask)
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        return gray

    def find_contours(self, gray, kernel_size=1):
        def create_square_kernel(size):
            if size % 2 == 0:
                raise ValueError("La taille du noyau doit être un nombre impair.")
            return np.ones((size, size), np.uint8)

        kernel = create_square_kernel(kernel_size)
        dilated_image = cv2.dilate(gray, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sorted(contours, key=cv2.contourArea, reverse=True)

    def find_highest_point(self, contour):
        highest_point = min(contour, key=lambda p: p[0][1])
        return tuple(highest_point[0])

    def process_contours(self, contours, taille_contour_min=10):
        vertical_line_x = self.image_width // 2
        intersection_point = None
        distance = float('inf')
        distance_min = 80
        highest_points = []

        def find_intersection(contour, x_coord):
            for point in contour:
                if point[0][0] == x_coord:
                    return tuple(point[0])
            return None

        for i, contour in enumerate(contours[:2]):
            if len(contour) > taille_contour_min:
                if self.plot:
                    self.plotter.plot_contour(contour)

                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                    highest_point = (cX, cY)
                    highest_points.append(highest_point)
                    if self.plot:
                        self.plotter.scatter_point(cX, cY, label=f'Point {i+1}' if i == 0 else "")

            intersection = find_intersection(contour, vertical_line_x)
            if intersection:
                intersection_point = intersection
                distance = self.image_height - intersection_point[1]
                if self.plot:
                    self.plotter.scatter_point(*intersection_point, label='Intersection')

            if self.plot:
                self.plotter.add_vertical_line(vertical_line_x)

            if intersection_point and self.plot:
                self.plotter.plot_line(vertical_line_x, self.image_height, vertical_line_x, intersection_point[1], 'r')
                self.distance = (self.image_height + intersection_point[1]) / 2
                self.plotter.add_text(vertical_line_x + 10, self.distance, f'Distance: {distance:.2f}')

        return highest_points, intersection_point, distance

    def calculate_target(self, highest_points, intersection_point, distance, distance_min=60):
        if len(highest_points) == 2:
            cX1, cY1 = highest_points[0]
            cX2, cY2 = highest_points[1]

            self.xd = (cX1 + cX2) / 2
            self.yd = (cY1 + cY2) / 2
            self.plotter.scatter_point(self.xd, self.yd, 'red', label='Target point')
            delta_theta = 0

        elif len(highest_points) > 0:
            self.xd, self.yd = highest_points[0]
            #self.xd -= 40
            #self.yd -= 30
            #if distance < distance_min:
            delta_theta = np.deg2rad(-20)
            #else:
            #delta_theta = 0
        else:
            self.steering_values.append(0.0)
            self.throttle_values.append(0.0)
            return 0

        x_ref = self.image_width // 2
        y_ref = self.image_height

        dx = self.xd - x_ref
        dy = y_ref - self.yd
        r = np.sqrt(dx**2 + dy**2)

        self.theta = math.atan2(dx, dy)
        self.theta = np.clip(self.theta, -np.pi / 2, np.pi / 2)
        theta_deg = np.rad2deg(self.theta)

        if self.plot:
            self.plotter.plot_line(x_ref, y_ref, self.xd, self.yd, 'r')
            self.plotter.add_text((x_ref + self.xd) / 2, (y_ref + self.yd) / 2, f'theta: {theta_deg:.2f}°')

        return delta_theta

    def adjust_theta(self, delta_theta, contours_white, intersection_point):

        x_ref = self.image_width // 2
        y_ref = self.image_height

        if delta_theta != 0.0:
            #if contours_white:
            #    contour_white = contours_white[0]
            #    contour_points_white = np.array([point[0] for point in contour_white])
            #    if len(contour_points_white) >= 2:
            #        [vx, vy, x, y] = cv2.fitLine(contour_points_white, cv2.DIST_L2, 0, 0.01, 0.01)
            #        slope = vy / vx
            #        intercept = y - slope * x
            #        distance_beteween_contours = abs(slope * self.xd - self.yd + intercept) / np.sqrt(slope**2 + 1)
            #        self.distance_beteween_contours = distance_beteween_contours[0]
#
            #    distance = self.process_contours(contours_white)[2]
            #    self.theta = -np.arcsin(np.clip(self.distance_beteween_contours / (2 * distance), -1, 1))
            #    self.theta = np.clip(self.theta, -np.pi/2, np.pi/2)
            #    print(11111111111111111111111111111111111111111111111111111111111111111111111111111111)
            #    print(self.theta)

            if intersection_point:
                dx_val = 120 - intersection_point[0] + 40
                dy_val = intersection_point[1]
                self.theta = -np.arctan(dx_val/dy_val)
                r = np.sqrt(dx_val**2 + dy_val**2)
                self.xd = r * np.sin(self.theta) + x_ref
                self.yd = -r * np.cos(self.theta) + y_ref

            else:
                dx_val = 40
                dy_val = self.distance
                self.theta = -np.arctan(dx_val/dy_val)
                r = np.sqrt(dx_val**2 + dy_val**2)
                self.xd = r * np.sin(self.theta) + x_ref
                self.yd = -r * np.cos(self.theta) + y_ref

            if self.plot:
                self.plotter.scatter_point(self.xd, self.yd, color='green', label='Midpoint')
                self.plotter.plot_line(80, 120, self.xd, self.yd, 'g')
                self.plotter.add_text((80 + self.xd) / 2, (120 + self.yd) / 2, f'theta new: {np.rad2deg(self.theta):.2f}°')
                self.theta_values.append(self.theta)
                self.plotter.show_legend()

    def calculate_derivatives(self):
        dt = 0.05
        dxd = (self.xd - self.xd_prev_prev) / (2 * dt)
        dyd = (self.yd - self.yd_prev_prev) / (2 * dt)
        ddxd = (self.xd - 2 * self.xd_prev + self.xd_prev_prev) / (dt**2)
        ddyd = (self.yd - 2 * self.yd_prev + self.yd_prev_prev) / (dt**2)

        self.xd_prev_prev = self.xd_prev
        self.xd_prev = self.xd

        self.yd_prev_prev = self.yd_prev
        self.yd_prev = self.yd

        return dxd, dyd, ddxd, ddyd

    def process_image(self):
        if self.image is None or self.image.size == 0:
            logging.warning("Empty image receipved!")
            self.steering_values.append(0.0)
            self.throttle_values.append(0.0)
            return 0.0, 0.0

        self.plotter.display_image(self.image)
        self.preprocess_image()

        gray_masked = self.apply_mask(0, 130, 145, 25, 255, 255)
        contours = self.find_contours(gray_masked, kernel_size=1)

        gray_white = self.apply_mask(0, 0, 200, 180, 50, 255)
        contours_white = self.find_contours(gray_white, kernel_size=1)

        highest_points, intersection_point, distance = self.process_contours(contours)

        delta_theta = self.calculate_target(highest_points, intersection_point, distance)

        self.adjust_theta(delta_theta, contours_white, intersection_point)

        dxd, dyd, ddxd, ddyd = self.calculate_derivatives()

        #ref = self.mpc.construct_reference_trajectory(self.xd, self.yd, dxd, dyd, ddxd, ddyd, self.theta)
        #ref = self.mpc.generate_curved_trajectory(80, 120, self.xd, self.yd, self.theta, N=50)
        #ref = self.mpc.generate_curved_trajectory(80, 120, self.xd, self.yd)
        #self.plotter.draw_trajectory(ref)

        self.plotter.update_plot()

        return self.xd, self.yd, dxd, dyd, ddxd, ddyd, self.theta



