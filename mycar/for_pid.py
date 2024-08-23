#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car

Usage:
    manage.py (drive) [--model=<model>] [--js] [--type=(linear|categorical)] [--camera=(single|stereo)] [--meta=<key:value> ...] [--myconfig=<filename>]
    manage.py (train) [--tubs=tubs] (--model=<model>) [--type=(linear|inferred|tensorrt_linear|tflite_linear)]

Options:
    -h --help               Show this screen.
    --js                    Use physical joystick.
    -f --file=<file>        A text file containing paths to tub files, one per line. Option may be used more than once.
    --meta=<key:value>      Key/Value strings describing describing a piece of meta data about this drive. Option may be used more than once.
    --myconfig=filename     Specify myconfig file to use. 
                            [default: myconfig.py]
"""
import os
import time
import logging
from docopt import docopt
import cv2
import matplotlib.pyplot as plt

import donkeycar as dk
from donkeycar.parts.transform import TriggeredCallback, DelayedTrigger
from donkeycar.parts.tub_v2 import TubWriter
#from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.controller import LocalWebController, JoystickController, WebFpv
from donkeycar.parts.throttle_filter import ThrottleFilter
from donkeycar.parts.behavior import BehaviorPart
from donkeycar.parts.file_watcher import FileWatcher
from donkeycar.parts.launch import AiLaunch
from donkeycar.parts.transform import Lambda
from donkeycar.utils import *

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)     

steering_values = []
throttle_values = []
errors = []
Y_values = []
theta_values = []

Y = np.array([[0.0], [0.0]])
theta = 0.0
throttle = 0.0
psi = 0.0
v = 0.01

xd_prev_prev = 0
yd_prev_prev = 0
xd_prev = 0
yd_prev = 0

distance_beteween_contours = 95

i_error = []
p_error = []
d_error = []


def drive(cfg, model_path=None, use_joystick=False, model_type=None,
          camera_type='single', meta=[]):
    global steering_values, throttle_values, errors, Y_values, theta_values, Y, theta, throttle, psi, v, xd_prev, yd_prev, xd_prev_prev, yd_prev_prev, distance_beteween_contours, i_error, p_error, d_error
    """
    Construct a working robotic vehicle from many parts. Each part runs as a
    job in the Vehicle loop, calling either it's run or run_threaded method
    depending on the constructor flag `threaded`. All parts are updated one
    after another at the framerate given in cfg.DRIVE_LOOP_HZ assuming each
    part finishes processing in a timely manner. Parts may have named outputs
    and inputs. The framework handles passing named outputs to parts
    requesting the same named input.
    """
    logger.info(f'PID: {os.getpid()}')
    if cfg.DONKEY_GYM:
        #the simulator will use cuda and then we usually run out of resources
        #if we also try to use cuda. so disable for donkey_gym.
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    if model_type is None:
        if cfg.TRAIN_LOCALIZER:
            model_type = "localizer"
        elif cfg.TRAIN_BEHAVIORS:
            model_type = "behavior"
        else:
            model_type = cfg.DEFAULT_MODEL_TYPE

    #Initialize car
    V = dk.vehicle.Vehicle()

    #Initialize logging before anything else to allow console logging
    if cfg.HAVE_CONSOLE_LOGGING:
        logger.setLevel(logging.getLevelName(cfg.LOGGING_LEVEL))
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(cfg.LOGGING_FORMAT))
        logger.addHandler(ch)

    if cfg.HAVE_MQTT_TELEMETRY:
        from donkeycar.parts.telemetry import MqttTelemetry
        tel = MqttTelemetry(cfg)

    if cfg.HAVE_ODOM:
        if cfg.ENCODER_TYPE == "GPIO":
            from donkeycar.parts.encoder import RotaryEncoder
            enc = RotaryEncoder(mm_per_tick=0.306096, pin = cfg.ODOM_PIN, debug = cfg.ODOM_DEBUG)
            V.add(enc, inputs=['throttle'], outputs=['enc/speed'], threaded=True)
        elif cfg.ENCODER_TYPE == "arduino":
            from donkeycar.parts.encoder import ArduinoEncoder
            enc = ArduinoEncoder()
            V.add(enc, outputs=['enc/speed'], threaded=True)
        else:
            print("No supported encoder found")

    logger.info("cfg.CAMERA_TYPE %s"%cfg.CAMERA_TYPE)
    if camera_type == "stereo":

        if cfg.CAMERA_TYPE == "WEBCAM":
            from donkeycar.parts.camera import Webcam

            camA = Webcam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam = 0)
            camB = Webcam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam = 1)

        elif cfg.CAMERA_TYPE == "CVCAM":
            from donkeycar.parts.cv import CvCam

            camA = CvCam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam = 0)
            camB = CvCam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam = 1)
        else:
            raise(Exception("Unsupported camera type: %s" % cfg.CAMERA_TYPE))

        V.add(camA, outputs=['cam/image_array_a'], threaded=True)
        V.add(camB, outputs=['cam/image_array_b'], threaded=True)

        from donkeycar.parts.image import StereoPair

        V.add(StereoPair(), inputs=['cam/image_array_a', 'cam/image_array_b'],
            outputs=['cam/image_array'])
    elif cfg.CAMERA_TYPE == "D435":
        from donkeycar.parts.realsense435i import RealSense435i
        cam = RealSense435i(
            enable_rgb=cfg.REALSENSE_D435_RGB,
            enable_depth=cfg.REALSENSE_D435_DEPTH,
            enable_imu=cfg.REALSENSE_D435_IMU,
            device_id=cfg.REALSENSE_D435_ID)
        V.add(cam, inputs=[],
              outputs=['cam/image_array', 'cam/depth_array',
                       'imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                       'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z'],
              threaded=True)

    else:
        if cfg.DONKEY_GYM:
            from donkeycar.parts.dgym import DonkeyGymEnv

        inputs = []
        outputs = ['cam/image_array']
        threaded = True
        if cfg.DONKEY_GYM:
            from donkeycar.parts.dgym import DonkeyGymEnv 
            #rbx
            cam = DonkeyGymEnv(cfg.DONKEY_SIM_PATH, host=cfg.SIM_HOST, env_name=cfg.DONKEY_GYM_ENV_NAME, conf=cfg.GYM_CONF, record_location=cfg.SIM_RECORD_LOCATION, record_gyroaccel=cfg.SIM_RECORD_GYROACCEL, record_velocity=cfg.SIM_RECORD_VELOCITY, delay=cfg.SIM_ARTIFICIAL_LATENCY)
            threaded = True
            inputs  = ['angle', 'throttle']
        elif cfg.CAMERA_TYPE == "PICAM":
            from donkeycar.parts.camera import PiCamera
            cam = PiCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, vflip=cfg.CAMERA_VFLIP, hflip=cfg.CAMERA_HFLIP)
        elif cfg.CAMERA_TYPE == "WEBCAM":
            from donkeycar.parts.camera import Webcam
            cam = Webcam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
        elif cfg.CAMERA_TYPE == "CVCAM":
            from donkeycar.parts.cv import CvCam
            cam = CvCam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
        elif cfg.CAMERA_TYPE == "CSIC":
            from donkeycar.parts.camera import CSICamera
            cam = CSICamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, framerate=cfg.CAMERA_FRAMERATE, gstreamer_flip=cfg.CSIC_CAM_GSTREAMER_FLIP_PARM)
        elif cfg.CAMERA_TYPE == "V4L":
            from donkeycar.parts.camera import V4LCamera
            cam = V4LCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, framerate=cfg.CAMERA_FRAMERATE)
        elif cfg.CAMERA_TYPE == "MOCK":
            from donkeycar.parts.camera import MockCamera
            cam = MockCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
        elif cfg.CAMERA_TYPE == "IMAGE_LIST":
            from donkeycar.parts.camera import ImageListCamera
            cam = ImageListCamera(path_mask=cfg.PATH_MASK)
        elif cfg.CAMERA_TYPE == "LEOPARD":
            from donkeycar.parts.leopard_imaging import LICamera
            cam = LICamera(width=cfg.IMAGE_W, height=cfg.IMAGE_H, fps=cfg.CAMERA_FRAMERATE)
        else:
            raise(Exception("Unkown camera type: %s" % cfg.CAMERA_TYPE))

        # add lidar
        if cfg.USE_LIDAR:
            from donkeycar.parts.lidar import RPLidar
            if cfg.LIDAR_TYPE == 'RP':
                print("adding RP lidar part")
                lidar = RPLidar(lower_limit = cfg.LIDAR_LOWER_LIMIT, upper_limit = cfg.LIDAR_UPPER_LIMIT)
                V.add(lidar, inputs=[],outputs=['lidar/dist_array'], threaded=True)
            if cfg.LIDAR_TYPE == 'YD':
                print("YD Lidar not yet supported")

        # Donkey gym part will output position information if it is configured
        if cfg.DONKEY_GYM:
            if cfg.SIM_RECORD_LOCATION:
                outputs += ['pos/pos_x', 'pos/pos_y', 'pos/pos_z', 'pos/speed', 'pos/cte']
            if cfg.SIM_RECORD_GYROACCEL:
                outputs += ['gyro/gyro_x', 'gyro/gyro_y', 'gyro/gyro_z', 'accel/accel_x', 'accel/accel_y', 'accel/accel_z']
            if cfg.SIM_RECORD_VELOCITY:
                outputs += ['vel/vel_x', 'vel/vel_y', 'vel/vel_z']
            
        V.add(cam, inputs=inputs, outputs=outputs, threaded=threaded)

    #This web controller will create a web server that is capable
    #of managing steering, throttle, and modes, and more.
    ctr = LocalWebController(port=cfg.WEB_CONTROL_PORT, mode=cfg.WEB_INIT_MODE)
    
    V.add(ctr,
        inputs=['cam/image_array', 'tub/num_records'],
        outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
        threaded=True)
        
    if use_joystick or cfg.USE_JOYSTICK_AS_DEFAULT:
        #modify max_throttle closer to 1.0 to have more power
        #modify steering_scale lower than 1.0 to have less responsive steering
        if cfg.CONTROLLER_TYPE == "MM1":
            from donkeycar.parts.robohat import RoboHATController            
            ctr = RoboHATController(cfg)
        elif "custom" == cfg.CONTROLLER_TYPE:
            #
            # custom controller created with `donkey createjs` command
            #
            from my_joystick import MyJoystickController
            ctr = MyJoystickController(
                throttle_dir=cfg.JOYSTICK_THROTTLE_DIR,
                throttle_scale=cfg.JOYSTICK_MAX_THROTTLE,
                steering_scale=cfg.JOYSTICK_STEERING_SCALE,
                auto_record_on_throttle=cfg.AUTO_RECORD_ON_THROTTLE)
            ctr.set_deadzone(cfg.JOYSTICK_DEADZONE)
        else:
            from donkeycar.parts.controller import get_js_controller

            ctr = get_js_controller(cfg)

            if cfg.USE_NETWORKED_JS:
                from donkeycar.parts.controller import JoyStickSub
                netwkJs = JoyStickSub(cfg.NETWORK_JS_SERVER_IP)
                V.add(netwkJs, threaded=True)
                ctr.js = netwkJs
        
        V.add(ctr, 
          inputs=['cam/image_array'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
          threaded=True)

    #this throttle filter will allow one tap back for esc reverse
    th_filter = ThrottleFilter()
    V.add(th_filter, inputs=['user/throttle'], outputs=['user/throttle'])


    #See if we should even run the pilot module.
    #This is only needed because the part run_condition only accepts boolean
    class PilotCondition:
        def run(self, mode):
            if mode == 'user':
                return False
            else:
                return True

    V.add(PilotCondition(), inputs=['user/mode'], outputs=['run_pilot'])

    class LedConditionLogic:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self, mode, recording, recording_alert, behavior_state, model_file_changed, track_loc):
            #returns a blink rate. 0 for off. -1 for on. positive for rate.

            if track_loc is not None:
                led.set_rgb(*self.cfg.LOC_COLORS[track_loc])
                return -1

            if model_file_changed:
                led.set_rgb(self.cfg.MODEL_RELOADED_LED_R, self.cfg.MODEL_RELOADED_LED_G, self.cfg.MODEL_RELOADED_LED_B)
                return 0.1
            else:
                led.set_rgb(self.cfg.LED_R, self.cfg.LED_G, self.cfg.LED_B)

            if recording_alert:
                led.set_rgb(*recording_alert)
                return self.cfg.REC_COUNT_ALERT_BLINK_RATE
            else:
                led.set_rgb(self.cfg.LED_R, self.cfg.LED_G, self.cfg.LED_B)

            if behavior_state is not None and model_type == 'behavior':
                r, g, b = self.cfg.BEHAVIOR_LED_COLORS[behavior_state]
                led.set_rgb(r, g, b)
                return -1 #solid on

            if recording:
                return -1 #solid on
            elif mode == 'user':
                return 1
            elif mode == 'local_angle':
                return 0.5
            elif mode == 'local':
                return 0.1
            return 0

    if cfg.HAVE_RGB_LED and not cfg.DONKEY_GYM:
        from donkeycar.parts.led_status import RGB_LED
        led = RGB_LED(cfg.LED_PIN_R, cfg.LED_PIN_G, cfg.LED_PIN_B, cfg.LED_INVERT)
        led.set_rgb(cfg.LED_R, cfg.LED_G, cfg.LED_B)

        V.add(LedConditionLogic(cfg), inputs=['user/mode', 'recording', "records/alert", 'behavior/state', 'modelfile/modified', "pilot/loc"],
              outputs=['led/blink_rate'])

        V.add(led, inputs=['led/blink_rate'])

    def get_record_alert_color(num_records):
        col = (0, 0, 0)
        for count, color in cfg.RECORD_ALERT_COLOR_ARR:
            if num_records >= count:
                col = color
        return col

    class RecordTracker:
        def __init__(self):
            self.last_num_rec_print = 0
            self.dur_alert = 0
            self.force_alert = 0

        def run(self, num_records):
            if num_records is None:
                return 0

            if self.last_num_rec_print != num_records or self.force_alert:
                self.last_num_rec_print = num_records

                if num_records % 10 == 0:
                    print("recorded", num_records, "records")

                if num_records % cfg.REC_COUNT_ALERT == 0 or self.force_alert:
                    self.dur_alert = num_records // cfg.REC_COUNT_ALERT * cfg.REC_COUNT_ALERT_CYC
                    self.force_alert = 0

            if self.dur_alert > 0:
                self.dur_alert -= 1

            if self.dur_alert != 0:
                return get_record_alert_color(num_records)

            return 0

    rec_tracker_part = RecordTracker()
    V.add(rec_tracker_part, inputs=["tub/num_records"], outputs=['records/alert'])

    if cfg.AUTO_RECORD_ON_THROTTLE and isinstance(ctr, JoystickController):
        #then we are not using the circle button. hijack that to force a record count indication
        def show_record_acount_status():
            rec_tracker_part.last_num_rec_print = 0
            rec_tracker_part.force_alert = 1
        ctr.set_button_down_trigger('circle', show_record_acount_status)

    #Sombrero
    if cfg.HAVE_SOMBRERO:
        from donkeycar.parts.sombrero import Sombrero
        s = Sombrero()

    #IMU
    if cfg.HAVE_IMU:
        from donkeycar.parts.imu import IMU
        imu = IMU(sensor=cfg.IMU_SENSOR, dlp_setting=cfg.IMU_DLP_CONFIG)
        V.add(imu, outputs=['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
            'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z'], threaded=True)

    # Use the FPV preview, which will show the cropped image output, or the full frame.
    if cfg.USE_FPV:
        V.add(WebFpv(), inputs=['cam/image_array'], threaded=True)

    #Behavioral state
    if cfg.TRAIN_BEHAVIORS:
        bh = BehaviorPart(cfg.BEHAVIOR_LIST)
        V.add(bh, outputs=['behavior/state', 'behavior/label', "behavior/one_hot_state_array"])
        try:
            ctr.set_button_down_trigger('L1', bh.increment_state)
        except:
            pass

        inputs = ['cam/image_array', "behavior/one_hot_state_array"]
    #IMU
    elif cfg.USE_LIDAR:
        inputs = ['cam/image_array', 'lidar/dist_array']

    elif cfg.HAVE_ODOM:
        inputs = ['cam/image_array', 'enc/speed']

    elif model_type == "imu":
        assert(cfg.HAVE_IMU)
        #Run the pilot if the mode is not user.
        inputs=['cam/image_array',
            'imu/acl_x', 'imu/acl_y', 'imu/acl_z',
            'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z']
    elif cfg.USE_LIDAR:
        inputs = ['cam/image_array', 'lidar/dist_array']
    else:
        inputs=['cam/image_array']

    def load_model(kl, model_path):
        start = time.time()
        print('loading model', model_path)
        kl.load(model_path)
        print('finished loading in %s sec.' % (str(time.time() - start)) )

    def load_weights(kl, weights_path):
        start = time.time()
        try:
            print('loading model weights', weights_path)
            kl.model.load_weights(weights_path)
            print('finished loading in %s sec.' % (str(time.time() - start)) )
        except Exception as e:
            print(e)
            print('ERR>> problems loading weights', weights_path)

    def load_model_json(kl, json_fnm):
        start = time.time()
        print('loading model json', json_fnm)
        from tensorflow.python import keras
        try:
            with open(json_fnm, 'r') as handle:
                contents = handle.read()
                kl.model = keras.models.model_from_json(contents)
            print('finished loading json in %s sec.' % (str(time.time() - start)) )
        except Exception as e:
            print(e)
            print("ERR>> problems loading model json", json_fnm)

    if model_path:
        #When we have a model, first create an appropriate Keras part
        kl = dk.utils.get_model_by_type(model_type, cfg)

        model_reload_cb = None

        if '.h5' in model_path or '.uff' in model_path or 'tflite' in model_path or '.pkl' in model_path:
            #when we have a .h5 extension
            #load everything from the model file
            load_model(kl, model_path)

            def reload_model(filename):
                load_model(kl, filename)

            model_reload_cb = reload_model

        elif '.json' in model_path:
            #when we have a .json extension
            #load the model from there and look for a matching
            #.wts file with just weights
            load_model_json(kl, model_path)
            weights_path = model_path.replace('.json', '.weights')
            load_weights(kl, weights_path)

            def reload_weights(filename):
                weights_path = filename.replace('.json', '.weights')
                load_weights(kl, weights_path)

            model_reload_cb = reload_weights

        else:
            print("ERR>> Unknown extension type on model file!!")
            return

        #this part will signal visual LED, if connected
        V.add(FileWatcher(model_path, verbose=True), outputs=['modelfile/modified'])

        #these parts will reload the model file, but only when ai is running so we don't interrupt user driving
        V.add(FileWatcher(model_path), outputs=['modelfile/dirty'], run_condition="ai_running")
        V.add(DelayedTrigger(100), inputs=['modelfile/dirty'], outputs=['modelfile/reload'], run_condition="ai_running")
        V.add(TriggeredCallback(model_path, model_reload_cb), inputs=["modelfile/reload"], run_condition="ai_running")

        outputs=['pilot/angle', 'pilot/throttle']

        if cfg.TRAIN_LOCALIZER:
            outputs.append("pilot/loc")

        V.add(kl, inputs=inputs,
              outputs=outputs,
              run_condition='run_pilot')
    
    if cfg.STOP_SIGN_DETECTOR:
        from donkeycar.parts.object_detector.stop_sign_detector import StopSignDetector
        V.add(StopSignDetector(cfg.STOP_SIGN_MIN_SCORE, cfg.STOP_SIGN_SHOW_BOUNDING_BOX), inputs=['cam/image_array', 'pilot/throttle'], outputs=['pilot/throttle', 'cam/image_array'])


    def process_image(image):
        global steering_values, throttle_values, errors, Y_values, theta_values, Y, theta, throttle, psi, v, xd_prev, yd_prev, xd_prev_prev, yd_prev_prev, distance_beteween_contours, i_error, p_error, d_error

        ############### TRAITEMENT DE L'IMAGE #################
        if image is None or image.size == 0:
            logging.warning("Empty image received!")
            steering_values.append(0.0)
            throttle_values.append(0.0)
            return 0.0, 0.0

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # RGB -> BGR
        image_height, image_width, _ = image.shape

        ##### PARAMETRES DU TRAITEMENT

        k = 3    # Taille du filtre de flou

        # Valeurs définies pour les seuils du masque
        lower_h = 0
        lower_s = 130
        lower_v = 145
        upper_h = 25
        upper_s = 255
        upper_v = 255

        # Taille du noyau
        kernel_size = 1

        ##### TRAITEMENT

        # Convertir l'image en HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Créer un masque pour la couleur
        lower_color = np.array([lower_h, lower_s, lower_v])
        upper_color = np.array([upper_h, upper_s, upper_v])
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Appliquer le masque à l'image originale
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # Convertir l'image masquée en niveaux de gris
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

        # Appliquer un filtre de flou
        if k > 1:
            gray = cv2.blur(gray, (k, k))

        # Créer un noyau carré rempli de 1
        def create_square_kernel(size):
            if size % 2 == 0:
                raise ValueError("La taille du noyau doit être un nombre impair.")
            return np.ones((size, size), np.uint8)

        kernel = create_square_kernel(kernel_size)

        # Appliquer dilatation pour combler les trous
        dilated_image = cv2.dilate(gray, kernel, iterations=1)

        # Trouver les contours
        contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Trier les contours par surface décroissante et garder les deux plus grands
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        ######### LIGNES BLANCHES
        lower_h = 0
        lower_s = 0
        lower_v = 200
        upper_h = 180
        upper_s = 50
        upper_v = 255

        # Créer un masque pour la couleur
        lower_color = np.array([lower_h, lower_s, lower_v])
        upper_color = np.array([upper_h, upper_s, upper_v])
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Appliquer le masque à l'image originale
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # Convertir l'image masquée en niveaux de gris
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

        # Appliquer un filtre de flou
        if k > 1:
            gray = cv2.blur(gray, (k, k))

        # Créer un noyau carré rempli de 1
        def create_square_kernel(size):
            if size % 2 == 0:
                raise ValueError("La taille du noyau doit être un nombre impair.")
            return np.ones((size, size), np.uint8)

        kernel = create_square_kernel(kernel_size)

        # Appliquer dilatation pour combler les trous
        dilated_image = cv2.dilate(gray, kernel, iterations=1)

        # Trouver les contours
        contours_white, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Trier les contours par surface décroissante et garder les deux plus grands
        contour_white = sorted(contours_white, key=cv2.contourArea, reverse=True)[:1]

                # Tracer la ligne de la distance sur l'image
                #y_closest = int(slope * x_d + intercept)
                #plt.figure(figsize=(10, 10))
                #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                #plt.plot([x_d, x_d], [y_d, y_closest], 'r-', linewidth=2)
                #plt.scatter([x_d], [y_d], color='blue')  # Tracer le point (xd, yd)
                #plt.plot([0, cols], [lefty, righty], 'g-', linewidth=2)  # Tracer la ligne ajustée

        ##### POINT OBJECTIF
        # Fonction pour trouver le point le plus haut d'un contour
        def find_highest_point(contour):
            # Trouver le point avec la plus petite coordonnée y
            highest_point = min(contour, key=lambda p: p[0][1])
            return tuple(highest_point[0])


        # Convertir l'image en RGB pour Matplotlib
        display_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Affichage avec Matplotlib
        #plt.figure(figsize=(10, 7))
        plt.clf()
        plt.imshow(display_image_rgb)
        plt.axis('off')  # Ne pas afficher les axes

        vertical_line_x = image_width // 2
        intersection_point = None
        distance = float('inf')
        distance_min = 60

        taille_contour_min = 30 #40

        highest_points = []

        def find_intersection(contour, x_coord):
            for point in contour:
                if point[0][0] == x_coord:
                    return tuple(point[0])
            return None

        for i, contour in enumerate(contours):
            if len(contour) > taille_contour_min:
                # Dessiner le contour
                contour_rgb = contour.reshape(-1, 2)
                plt.plot(contour_rgb[:, 0], contour_rgb[:, 1], 'g-', linewidth=2)

                #contour = contour + [0, image_height]
                #plt.plot(contour[:, 0, 0], contour[:, 0, 1], 'g-', linewidth=2)
                
                use_moments = True
                use_highest_point = False
                # Trouver le point le plus haut
                if use_moments:
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cX = int(M['m10'] / M['m00'])
                        cY = int(M['m01'] / M['m00'])
                        highest_point = (cX, cY)
                        highest_points.append(highest_point)
                        plt.scatter(*highest_point, color='blue', s=40, edgecolor='black', label=f'Point {i+1}' if i == 0 else "")

                elif use_highest_point:
                    highest_point = find_highest_point(contour)
                    highest_points.append(highest_point)
                    plt.scatter(*highest_point, color='blue', s=40, edgecolor='black', label=f'Point {i+1}' if i == 0 else "")
                    
            # Trouver l'intersection avec la ligne verticale
            intersection = find_intersection(contour, vertical_line_x)
            if intersection:
                intersection_point = intersection
                distance = image_height - intersection_point[1]
                plt.scatter(*intersection_point, color='blue', s=40, edgecolor='black', label='Intersection')

            # Dessiner la ligne verticale
            plt.axvline(x=vertical_line_x, color='yellow', linestyle='--', label='Vertical Line')

            if intersection_point:
                # Dessiner la distance
                plt.plot([vertical_line_x, vertical_line_x], [image_height, intersection_point[1]], 'r--')
                plt.text(vertical_line_x + 10, (image_height + intersection_point[1]) / 2, f'Distance: {distance:.2f}', color='white')

        if len(highest_points) == 2 and len(contours[0]) > taille_contour_min and len(contours[1]) > taille_contour_min:
            cX1, cY1 = highest_points[0]
            cX2, cY2 = highest_points[1]

            xd = (cX1 + cX2) / 2
            yd = (cY1 + cY2) / 2

            distance_beteween_contours = np.sqrt((cX2 - cX1) ** 2 + (cY2 - cY1) ** 2)
            #print(f"La distance entre les deux contours est : {distance_beteween_contours:.2f} pixels")

            plt.scatter(xd, yd, color='red', s=40, label='Midpoint', edgecolor='black')

            delta_theta = 0

        elif len(highest_points)>0:
            #print('!!!!!! Only one contour detected !!!!!!!!!')
            xd, yd = highest_points[0]
            #print('distance : ', distance)
            if distance < distance_min:
                #if theta < 0:
                delta_theta = np.deg2rad(-20) # -20
                #else:
                    #delta_theta = np.deg2rad(90)
            else:
                delta_theta = 0
            #return 0.0, 0.0
        else:
            steering_values.append(0.0)
            throttle_values.append(0.0)
            return 0.0, 0.0

        x_ref = image_width // 2
        y_ref = image_height

        dx = xd - x_ref
        dy = y_ref - yd
        r = np.sqrt(dx**2 + dy**2)

        theta = math.atan2(dx, dy)
        #print('theta before clip : ', theta)
        theta = np.clip(theta, -np.pi / 2, np.pi / 2)
        #print('theta after clip : ', theta)

        theta_deg = np.rad2deg(theta)
        #print('theta : ', theta_deg)

        plt.plot([x_ref, xd], [y_ref, yd], 'r-', linewidth=2)
        plt.text((x_ref + xd) / 2, (y_ref + yd) / 2, f'theta: {theta_deg:.2f}°', color='red', fontsize=12)

        def sawtooth(x):
            return (x+np.pi)%(2*np.pi)-np.pi   # or equivalently   2*arctan(tan(x/2))

        #theta = sawtooth(theta) 

        if delta_theta != 0.0:
            #print('theta before delta_theta : ', np.rad2deg(theta))
            #theta = theta + delta_theta

            #print(distance_beteween_contours/(2*distance))

            if contour_white:
                # Extraire les points du contour et convertir en tableau NumPy
                contour_white = contour_white[0]
                contour_points_white = np.array([point[0] for point in contour_white])  # Convertir en tableau NumPy
                
                # Vérifier qu'il y a suffisamment de points pour ajuster une droite
                if len(contour_points_white) >= 2:
                    # Appliquer la régression linéaire pour ajuster une droite
                    [vx, vy, x, y] = cv2.fitLine(contour_points_white, cv2.DIST_L2, 0, 0.01, 0.01)
                    slope = vy / vx
                    intercept = y - slope * x

                    # Calculer les points de la ligne pour tracer
                    rows, cols = image.shape[:2]
                    lefty = int(slope * 0 + intercept)
                    righty = int(slope * cols + intercept)

                    #plt.plot([0, image_width - 1], [lefty, righty], 'r-', linewidth=2, label='Fitted Line')

                    # Tracer la ligne sur l'image
                    #cv2.line(image, (0, lefty), (cols - 1, righty), (0, 255, 0), 2)

                    # Afficher l'image avec la ligne tracée
                    #cv2.imshow('Image with Fitted Line', image)
                    #cv2.waitKey(1)
                    #cv2.destroyAllWindows()
                    
                    # Point donné (xd, yd)
                    #x_d = 150  # à remplacer par votre valeur
                    #y_d = 200  # à remplacer par votre valeur
                    
                    # Calculer la distance entre la ligne et le point (xd, yd)
                    distance_beteween_contours = abs(slope * xd - yd + intercept) / np.sqrt(slope**2 + 1)
                    distance_beteween_contours = distance_beteween_contours[0]
                    #print(distance)
                    print(f"La distance entre la ligne et le point ({xd}, {yd}) est : {distance_beteween_contours:.2f} pixels")
                    

            theta = -np.arcsin(np.clip(distance_beteween_contours/(2*distance), -1, 1)) - 1
            if theta > 0:
                print('ISSUE !!!!!!!!!!!!')
            #print('intermediate theta : ', np.rad2deg(theta))
            theta = np.clip(theta, -np.pi/2, np.pi/2)

            dx = intersection_point[0] - x_ref
            dy = y_ref - intersection_point[1]

            r = np.sqrt(dx**2 + dy**2)

            xd = r * np.sin(theta) + x_ref
            yd = -r * np.cos(theta) + y_ref

            #print('theta : ', np.rad2deg(theta))

            plt.scatter(xd, yd, color='green', s=40, label='Midpoint', edgecolor='black')

            plt.plot([x_ref, xd], [y_ref, yd], 'g-', linewidth=2)
            plt.text((x_ref + xd) / 2, (y_ref + yd) / 2, f'theta new: {np.rad2deg(theta):.2f}°', color='green', fontsize=12)

        theta_values.append(theta)

        plt.legend()
        plt.draw()
        plt.pause(0.001)  # Pause pour l'affichage en temps réel


        ########## CONTROLER ############
        dt = 0.05
        L = 0.01

        def ctrl_FBLN(Y, dY, W, dW, ddW, theta, psi, v, L, r, ddxd, ddyd): 
            global errors
            y1, y2 = Y.flatten()

            # cos et sin inverses
            A = np.array([[-np.sin(theta), -r*v*np.cos(theta)/L], 
                          [np.cos(theta), r*v*np.sin(theta)/L]])

            B = np.array([[-2*(v**2)*psi*np.cos(theta)/L + (r*v**2)*(psi**2)*np.sin(theta)/(L**2) + ddxd], 
                          [-2*(v**2)*psi*np.sin(theta)/L - (r*v**2)*(psi**2)*np.cos(theta)/(L**2) + ddyd]])
            
            ki = 1
            kp = 5#10#20
            kd = 1
            #print('ki error : ', W-Y)
            #print('kp error : ', dW-dY)
            #print('kd error : ', ddW)
            i_error.append(W-Y)
            p_error.append(dW-dY)
            d_error.append(ddW)
            V = ki*(W-Y) + kp*(dW-dY) + kd*ddW

            errors.append(ki*(W-Y) + kp*(dW-dY) + kd*(ddW-V))

            U = np.linalg.inv(A) @ (V - B)
            return U

        def f(Y, theta, psi, L, v, r, dxd, dyd): 
            y1, y2 = Y.flatten()
            dy1 = -v*np.sin(theta) - (r*v*psi*np.cos(theta))/L + dxd
            dy2 = v*np.cos(theta) - (r*v*psi*np.sin(theta))/L + dyd
            return np.array([[dy1],[dy2]])

        dxd = (xd - xd_prev_prev) / (2*dt)
        dyd = (yd - yd_prev_prev) / (2*dt)
        ddxd = (xd - 2*xd_prev + xd_prev_prev) / (dt**2)
        ddyd = (yd - 2*yd_prev + yd_prev_prev) / (dt**2)

        xd_prev_prev = xd_prev
        xd_prev = xd


        dY = f(Y, theta, psi, L, v, r, dxd, dyd)
        Y = Y + dY * dt
        Y_values.append(Y)

        W = np.array([[xd], [yd]])
        dW = np.array([[dxd], [dyd]])
        #dW = np.array([[0], [0]])
        ddW = np.array([[ddxd], [ddyd]]) # hyp v' = psi' = 0

        U = ctrl_FBLN(Y, dY, W, dW, ddW, theta, psi, v, L, r, ddxd, ddyd) # U = (u1, u2) = (v', psi')

        u1, u2 = U.flatten()
        v += u1 * dt
        #print('u2 : ', u2)
        psi += u2 * dt

        if psi < -np.pi:
            psi += np.pi
        elif psi > np.pi:
            psi += np.pi
        #psi = sawtooth(psi)
        #print('psi : ', psi)
        steering = np.clip(psi, -0.8, 0.8) #psi / (np.pi/2) # à vérif les valeurs extrêmes de psi
        #print('steering : ', steering)

        #print('v : ', v)
        v = np.clip(v, 0.01, 0.2)
        throttle = v #/ 2
        #
        #throttle = 0.05
        #print('throttle : ', throttle)

        throttle_values.append(throttle)
        steering_values.append(steering)

        #print('steering : ', steering)
        #print('throttle : ', throttle)

        return steering, throttle



        # Lambda part to process the images and find line position
    V.add(Lambda(process_image), inputs=['cam/image_array'], outputs=['user/angle', 'user/throttle'])

    #Choose what inputs should change the car.
    class DriveMode:
        def run(self, mode, user_angle, user_throttle, pilot_angle, pilot_throttle, param_line, error):
            if mode == 'user':
                return user_angle, user_throttle
            elif mode == 'local_angle':
                return pilot_angle if pilot_angle else 0.0, user_throttle
            else:
                return pilot_angle if pilot_angle else 0.0, pilot_throttle * self.cfg.AI_THROTTLE_MULT if pilot_throttle else 0.0

    V.add(DriveMode(),
          inputs=['user/mode', 'user/angle', 'user/throttle',
                  'pilot/angle', 'pilot/throttle', 'param_line', 'error'],
          outputs=['angle', 'throttle'])


    #to give the car a boost when starting ai mode in a race.
    aiLauncher = AiLaunch(cfg.AI_LAUNCH_DURATION, cfg.AI_LAUNCH_THROTTLE, cfg.AI_LAUNCH_KEEP_ENABLED)

    V.add(aiLauncher,
        inputs=['user/mode', 'throttle'],
        outputs=['throttle'])

    if isinstance(ctr, JoystickController):
        ctr.set_button_down_trigger(cfg.AI_LAUNCH_ENABLE_BUTTON, aiLauncher.enable_ai_launch)


    class AiRunCondition:
        '''
        A bool part to let us know when ai is running.
        '''
        def run(self, mode):
            if mode == "user":
                return False
            return True

    V.add(AiRunCondition(), inputs=['user/mode'], outputs=['ai_running'])

    #Ai Recording
    class AiRecordingCondition:
        '''
        return True when ai mode, otherwize respect user mode recording flag
        '''
        def run(self, mode, recording):
            if mode == 'user':
                return recording
            return True

    if cfg.RECORD_DURING_AI:
        V.add(AiRecordingCondition(), inputs=['user/mode', 'recording'], outputs=['recording'])

    #Drive train setup
    if cfg.DONKEY_GYM or cfg.DRIVE_TRAIN_TYPE == "MOCK":
        pass
    elif cfg.DRIVE_TRAIN_TYPE == "SERVO_ESC":
        from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle

        steering_controller = PCA9685(cfg.STEERING_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
        steering = PWMSteering(controller=steering_controller,
                                        left_pulse=cfg.STEERING_LEFT_PWM,
                                        right_pulse=cfg.STEERING_RIGHT_PWM)

        throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
        throttle = PWMThrottle(controller=throttle_controller,
                                        max_pulse=cfg.THROTTLE_FORWARD_PWM,
                                        zero_pulse=cfg.THROTTLE_STOPPED_PWM,
                                        min_pulse=cfg.THROTTLE_REVERSE_PWM)

        V.add(steering, inputs=['angle'], threaded=True)
        V.add(throttle, inputs=['throttle'], threaded=True)


    elif cfg.DRIVE_TRAIN_TYPE == "DC_STEER_THROTTLE":
        from donkeycar.parts.actuator import Mini_HBridge_DC_Motor_PWM

        steering = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_LEFT, cfg.HBRIDGE_PIN_RIGHT)
        throttle = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_FWD, cfg.HBRIDGE_PIN_BWD)

        V.add(steering, inputs=['angle'])
        V.add(throttle, inputs=['throttle'])


    elif cfg.DRIVE_TRAIN_TYPE == "DC_TWO_WHEEL":
        from donkeycar.parts.actuator import TwoWheelSteeringThrottle, Mini_HBridge_DC_Motor_PWM

        left_motor = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_LEFT_FWD, cfg.HBRIDGE_PIN_LEFT_BWD)
        right_motor = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_RIGHT_FWD, cfg.HBRIDGE_PIN_RIGHT_BWD)
        two_wheel_control = TwoWheelSteeringThrottle()

        V.add(two_wheel_control,
                inputs=['throttle', 'angle'],
                outputs=['left_motor_speed', 'right_motor_speed'])

        V.add(left_motor, inputs=['left_motor_speed'])
        V.add(right_motor, inputs=['right_motor_speed'])

    elif cfg.DRIVE_TRAIN_TYPE == "DC_TWO_WHEEL_L298N":
        from donkeycar.parts.actuator import TwoWheelSteeringThrottle, L298N_HBridge_DC_Motor

        left_motor = L298N_HBridge_DC_Motor(cfg.HBRIDGE_L298N_PIN_LEFT_FWD, cfg.HBRIDGE_L298N_PIN_LEFT_BWD, cfg.HBRIDGE_L298N_PIN_LEFT_EN)
        right_motor = L298N_HBridge_DC_Motor(cfg.HBRIDGE_L298N_PIN_RIGHT_FWD, cfg.HBRIDGE_L298N_PIN_RIGHT_BWD, cfg.HBRIDGE_L298N_PIN_RIGHT_EN)
        two_wheel_control = TwoWheelSteeringThrottle()

        V.add(two_wheel_control,
                inputs=['throttle', 'angle'],
                outputs=['left_motor_speed', 'right_motor_speed'])

        V.add(left_motor, inputs=['left_motor_speed'])
        V.add(right_motor, inputs=['right_motor_speed'])


    elif cfg.DRIVE_TRAIN_TYPE == "SERVO_HBRIDGE_PWM":
        from donkeycar.parts.actuator import ServoBlaster, PWMSteering
        steering_controller = ServoBlaster(cfg.STEERING_CHANNEL) #really pin
        #PWM pulse values should be in the range of 100 to 200
        assert(cfg.STEERING_LEFT_PWM <= 200)
        assert(cfg.STEERING_RIGHT_PWM <= 200)
        steering = PWMSteering(controller=steering_controller,
                                        left_pulse=cfg.STEERING_LEFT_PWM,
                                        right_pulse=cfg.STEERING_RIGHT_PWM)


        from donkeycar.parts.actuator import Mini_HBridge_DC_Motor_PWM
        motor = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_FWD, cfg.HBRIDGE_PIN_BWD)

        V.add(steering, inputs=['angle'], threaded=True)
        V.add(motor, inputs=["throttle"])
        
    elif cfg.DRIVE_TRAIN_TYPE == "MM1":
        from donkeycar.parts.robohat import RoboHATDriver
        V.add(RoboHATDriver(cfg), inputs=['angle', 'throttle'])
    
    elif cfg.DRIVE_TRAIN_TYPE == "PIGPIO_PWM":
        from donkeycar.parts.actuator import PWMSteering, PWMThrottle, PiGPIO_PWM
        steering_controller = PiGPIO_PWM(cfg.STEERING_PWM_PIN, freq=cfg.STEERING_PWM_FREQ, inverted=cfg.STEERING_PWM_INVERTED)
        steering = PWMSteering(controller=steering_controller,
                                        left_pulse=cfg.STEERING_LEFT_PWM, 
                                        right_pulse=cfg.STEERING_RIGHT_PWM)
        
        throttle_controller = PiGPIO_PWM(cfg.THROTTLE_PWM_PIN, freq=cfg.THROTTLE_PWM_FREQ, inverted=cfg.THROTTLE_PWM_INVERTED)
        throttle = PWMThrottle(controller=throttle_controller,
                                            max_pulse=cfg.THROTTLE_FORWARD_PWM,
                                            zero_pulse=cfg.THROTTLE_STOPPED_PWM, 
                                            min_pulse=cfg.THROTTLE_REVERSE_PWM)
        V.add(steering, inputs=['angle'], threaded=True)
        V.add(throttle, inputs=['throttle'], threaded=True)

    # OLED setup
    if cfg.USE_SSD1306_128_32:
        from donkeycar.parts.oled import OLEDPart
        auto_record_on_throttle = cfg.USE_JOYSTICK_AS_DEFAULT and cfg.AUTO_RECORD_ON_THROTTLE
        oled_part = OLEDPart(cfg.SSD1306_128_32_I2C_BUSNUM, auto_record_on_throttle=auto_record_on_throttle)
        V.add(oled_part, inputs=['recording', 'tub/num_records', 'user/mode'], outputs=[], threaded=True)

    #add tub to save data

    if cfg.USE_LIDAR:
        inputs = ['cam/image_array', 'lidar/dist_array', 'user/angle', 'user/throttle', 'user/mode']
        types = ['image_array', 'nparray','float', 'float', 'str']
    else:
        inputs=['cam/image_array','user/angle', 'user/throttle', 'user/mode']
        types=['image_array','float', 'float','str']

    if cfg.USE_LIDAR:
        inputs += ['lidar/dist_array']
        types += ['nparray']

    if cfg.HAVE_ODOM:
        inputs += ['enc/speed']
        types += ['float']

    if cfg.TRAIN_BEHAVIORS:
        inputs += ['behavior/state', 'behavior/label', "behavior/one_hot_state_array"]
        types += ['int', 'str', 'vector']

    if cfg.CAMERA_TYPE == "D435" and cfg.REALSENSE_D435_DEPTH:
        inputs += ['cam/depth_array']
        types += ['gray16_array']

    if cfg.HAVE_IMU or (cfg.CAMERA_TYPE == "D435" and cfg.REALSENSE_D435_IMU):
        inputs += ['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
            'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z']

        types +=['float', 'float', 'float',
           'float', 'float', 'float']

    # rbx
    if cfg.DONKEY_GYM:
        if cfg.SIM_RECORD_LOCATION:  
            inputs += ['pos/pos_x', 'pos/pos_y', 'pos/pos_z', 'pos/speed', 'pos/cte']
            types  += ['float', 'float', 'float', 'float', 'float']
        if cfg.SIM_RECORD_GYROACCEL: 
            inputs += ['gyro/gyro_x', 'gyro/gyro_y', 'gyro/gyro_z', 'accel/accel_x', 'accel/accel_y', 'accel/accel_z']
            types  += ['float', 'float', 'float', 'float', 'float', 'float']
        if cfg.SIM_RECORD_VELOCITY:  
            inputs += ['vel/vel_x', 'vel/vel_y', 'vel/vel_z']
            types  += ['float', 'float', 'float']

    if cfg.RECORD_DURING_AI:
        inputs += ['pilot/angle', 'pilot/throttle']
        types += ['float', 'float']

    if cfg.HAVE_PERFMON:
        from donkeycar.parts.perfmon import PerfMonitor
        mon = PerfMonitor(cfg)
        perfmon_outputs = ['perf/cpu', 'perf/mem', 'perf/freq']
        inputs += perfmon_outputs
        types += ['float', 'float', 'float']
        V.add(mon, inputs=[], outputs=perfmon_outputs, threaded=True)

    # do we want to store new records into own dir or append to existing
    tub_path = TubHandler(path=cfg.DATA_PATH).create_tub_path() if \
        cfg.AUTO_CREATE_NEW_TUB else cfg.DATA_PATH
    tub_writer = TubWriter(tub_path, inputs=inputs, types=types, metadata=meta)
    V.add(tub_writer, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')

    # Telemetry (we add the same metrics added to the TubHandler
    if cfg.HAVE_MQTT_TELEMETRY:
        telem_inputs, _ = tel.add_step_inputs(inputs, types)
        V.add(tel, inputs=telem_inputs, outputs=["tub/queue_size"], threaded=True)

    if cfg.PUB_CAMERA_IMAGES:
        from donkeycar.parts.network import TCPServeValue
        from donkeycar.parts.image import ImgArrToJpg
        pub = TCPServeValue("camera")
        V.add(ImgArrToJpg(), inputs=['cam/image_array'], outputs=['jpg/bin'])
        V.add(pub, inputs=['jpg/bin'])

    if type(ctr) is LocalWebController:
        if cfg.DONKEY_GYM:
            print("You can now go to http://localhost:%d to drive your car." % cfg.WEB_CONTROL_PORT)
        else:
            print("You can now go to <your hostname.local>:%d to drive your car." % cfg.WEB_CONTROL_PORT)
    elif isinstance(ctr, JoystickController):
        print("You can now move your joystick to drive your car.")
        ctr.set_tub(tub_writer.tub)
        ctr.print_controls()

    #V.add(Lambda(detect_lines_hough), inputs=['cam/image_array'], outputs=['param_line', 'error'])

    def plot_graphs():
        global steering_values, throttle_values, errors, Y_values, theta_values, Y, theta, throttle, psi, v, i_error, p_error, d_error, xd_prev, xd_prev_prev, yd_prev, yd_prev_prev

        print('Plotting graphs...')

        # 1. Orientation Angle (Theta) Over Time
        plt.figure(figsize=(10, 6))
        dt = 0.05  # exemple pour un échantillonnage à 10 Hz
        time_values = np.arange(len(theta_values)) * dt
        plt.plot(time_values, theta_values, color='green', label='Orientation angle (theta)')
        #plt.plot(theta_values, color='green', label='Orientation angle (theta)')
        plt.axhline(y=0, color='red', linestyle='--', label='Desired theta (0)')
        plt.xlabel('Time [s]')
        plt.ylabel('Theta [radians]')
        plt.title('Orientation angle (theta) over time')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 2. Steering Angle and Throttle Over Time
        plt.figure(figsize=(10, 6))
        plt.plot(steering_values, color='red', label='Steering Angle')
        plt.plot(throttle_values, color='blue', label='Throttle')
        plt.xlabel('Time [s]')
        plt.ylabel('Value')
        plt.title('Steering angle and throttle over time')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 3. X Error Over Time (Proportional, Integral, Derivative)
        error_x_p = np.array([error[0, 0] for error in p_error])
        error_x_i = np.array([error[0, 0] for error in i_error])
        error_x_d = np.array([error[0, 0] for error in d_error])

        # Filter the errors to include only values within the range of -10 to 10
        mask_p = np.abs(error_x_p)
        mask_i = np.abs(error_x_i)
        mask_d = np.abs(error_x_d)

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(error_x_p))[mask_p], error_x_p[mask_p], label='Proportional Error (X)', color='blue')
        plt.plot(np.arange(len(error_x_i))[mask_i], error_x_i[mask_i], label='Integral Error (X)', color='orange')
        plt.plot(np.arange(len(error_x_d))[mask_d], error_x_d[mask_d], label='Derivative Error (X)', color='red')
        plt.xlabel('Time [s]')
        plt.ylabel('X error [pixel]')
        plt.title('X direction errors over time')
        plt.legend()
        plt.grid(True)
        plt.show()

    def exit_handler():
        plot_graphs()
        #pass

    atexit.register(exit_handler)

    #run the vehicle for 20 seconds
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ, max_loop_count=cfg.MAX_LOOPS)


    V.add(Lambda(plot_graphs), inputs=['user/angle', 'user_throttle'], outputs=[])


if __name__ == '__main__':
    import atexit
    args = docopt(__doc__)
    cfg = dk.load_config(myconfig=args['--myconfig'])

    if args['drive']:
        model_type = args['--type']
        camera_type = args['--camera']
        drive(cfg, model_path=args['--model'], use_joystick=args['--js'],
              model_type=model_type, camera_type=camera_type,
              meta=args['--meta'])
    elif args['train']:
        print('Use python train.py instead.\n')
