# Raspberry Pi Data
- **IP Address:** 192.168.1.207
- **User:** pi  
- **Password:** ky”tefwu

# Downloading and Setting Up the Simulation

This project was developed on a MacOS M1 using Unity. It is likely to work on Linux as well. If there are issues, consider using Unity on Linux. The benefit of using Unity is that you can run the same code in both simulated and real environments.

## Steps to Download the Simulation

1. **Clone the Repository:**
    ```bash
    cd ~/projects
    git clone https://github.com/tawnkramer/gym-donkeycar
    cd gym-donkeycar
    conda activate donkey
    pip install -e .[gym-donkeycar]
    ```

2. **Set Up the Donkey Environment:**
    ```bash
    conda create -n donkey python=3.11
    conda activate donkey
    ```

3. **Install DonkeyCar:**
    - For PC:
        ```bash
        pip install donkeycar[pc]
        ```
    - For Mac:
        ```bash
        pip install donkeycar[macos]
        ```

4. **Create a New Car Project:**
    ```bash
    donkey createcar --path ~/mycar
    ```

5. **Download the Simulation:**
    - [Donkey Simulator Download Link](https://github.com/tawnkramer/gym-donkeycar/releases/)

6. **Install Unity:**
   - Download and install the Unity version corresponding to the simulation version (2020.3.4f1 was used for this project).

7. **Modify the Simulation in Unity:**
    - Open the simulation editor in Unity.
    - Make any desired modifications. For reinforcement learning, it is especially important to enable the side buttons:
        - Joystick/Keyboard No Rec
        - Joystick/Keyboard w Rec
        - Auto Drive w Rec
        - Auto Drive No Rec
        - NN Control over Network
    - Once satisfied with the modifications, build the simulation and place it in `sdsandbox/sdsim/`.

8. **Modify `myconfig.py` as Needed:**
    ```python
    DONKEY_GYM = True
    DONKEY_SIM_PATH = "/home/<user-name>/projects/DonkeySimLinux/donkey_sim.x86_64"
    DONKEY_GYM_ENV_NAME = "donkey-waveshare-v0"
    ```

# Useful Commands for Running the Simulation

1. **Start the Python Environment:**
    ```bash
    python3
    ```

2. **Initialize the Simulation Environment:**
    ```python
    import gym_donkeycar
    import gym
    env = gym.make('donkey-mountain-track-v0')
    ```

3. **Reset the Car:**
    ```python
    obs = env.reset()
    ```

4. **Display the Camera View in a Window:**
    ```python
    cv2.imshow('test', obs)
    cv2.waitKey(0)
    ```

5. **Convert BGR to RGB:**
    ```python
    cv2.imshow('test', obs[:,:,::-1])
    cv2.waitKey(-1)
    ```

6. **Reset the Circuit in the Car:**
    ```bash
    kill %1
    ```

7. **Simulate a Step:**
    ```python
    obs, reward, done, infos = env.step([a, b])
    ```
    - `a`: steering
    - `b`: throttle
    - `reward`: calculated based on the centerline.
    - `done`: status of the simulation (True/False).
    - `infos`: additional information.

8. **Close the Environment:**
    ```python
    env.close()
    ```

9. **Train a Model Using PPO:**
    ```python
    from stable_baselines3 import PPO
    model = PPO('CnnPolicy', env, verbose=1)
    model = PPO('CnnPolicy', env, n_steps=200, verbose=1)
    model.learn(10_000)
    ```

# Python Scripts

- **`manage.py`:** Use this script to launch the simulation or control the real car with various options.
    ```bash
    Usage:
    manage.py (drive) [--model=<model>] [--js] [--type=(linear|categorical)] [--camera=(single|stereo)] [--meta=<key:value> ...] [--myconfig=<filename>]
    manage.py (train) [--tubs=tubs] (--model=<model>) [--type=(linear|inferred|tensorrt_linear|tflite_linear)]
    ```

- **Additional Scripts:**
    - `python for_pid.py drive`
    - `python for_mpc.py drive`

# Usefull tips
To have a real-time view of what the camera captures and to control the car (both in simulation and the real car), once manage.py drive is launched, you can go to http://localhost:8887/drive if you are using the simulation, or http://192.168.1.207:8887/drive for the real car.

This is useful for choosing how to control the car or test a controller: you can either control the car with the keyboard, joystick, or use the controller designed to automate it. The Python scripts I made directly utilize the automation mode.

Another nice interface is the donkey ui one. It is particularly interesting when dealing with the reinforcement learninig. You access it directly in the console. The Donkey UI provides several useful windows for managing, training, and interacting with your car. Here’s a breakdown of each window:

- **Tub Manager**:
  - Helps you manage the data used for training.
  - You can delete poor-quality data.
  - Apply filters to refine the dataset.

- **Trainer**:
  - Used to train your model.
  - Configure training parameters and start the training process.
  - Monitor training progress and adjust as needed.

- **Pilot Arena**:
  - Displays how the model performs after training.
  - Visualize the model's behavior in different scenarios.

- **Car Connector**:
  - Allows interaction with the car.
  - Push data (such as a trained model) to the car.
  - Pull data (like images) from the car to your PC.
  - Workflow: Pull images from the car, train the model on your PC, then push the trained model back to the Raspberry Pi to test it.

# Issues Encountered

1. **Reinforcement Learning:**
    - When using `rl-baselines3-zoo` for reinforcement learning, do not use the `feat/offline-RL` branch as mentioned in some tutorials. Instead, use the `feat/gym-donkeycar` branch. An error you might encounter with the wrong branch is `gym.error.NameNotFound: Environment 'donkey-waveshare' doesn't exist.`

2. **Autoencoder:**
    - Use the master branch of `aae-train-donkeycar`.

3. **TFLite Conversion:**
    - To convert a `.keras` model to TFLite during reinforcement learning:
    ```bash
    !pip install tf-nightly
    !pip install -q --upgrade keras-nlp
    !pip install -q -U keras>=3
    pip install tf_keras
    pip install tf-nightly
    pip install -q --upgrade keras-nlp
    pip install -q -U keras>=3
    ```