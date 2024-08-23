import json

class TrajectoryLogger:
    def __init__(self, filename='trajectory_data.json'):
        self.filename = filename
        self.data = {
            'trajectory': [],
            'parameters': {}
        }
    
    def add_trajectory_point(self, x, y, theta, xd, yd, dxd, dyd, ddxd, ddyd, steering, throttle):
        """
        Ajoute un point de trajectoire avec les coordonnées x, y et l'orientation theta,
        ainsi que les paramètres de la dynamique du véhicule.
        """
        point = {
            'x': x,
            'y': y,
            'theta': theta,
            'xd': xd,
            'yd': yd,
            'dxd': dxd,
            'dyd': dyd,
            'ddxd': ddxd,
            'ddyd': ddyd,
            'steering': steering,
            'throttle': throttle
        }
        self.data['trajectory'].append(point)
    
    def set_parameters(self, **kwargs):
        """
        Enregistre les paramètres généraux utilisés pour la génération de la trajectoire.
        """
        self.data['parameters'] = kwargs
    
    def save_to_json(self):
        """
        Enregistre les données accumulées dans un fichier JSON.
        """
        with open(self.filename, 'w') as f:
            json.dump(self.data, f, indent=4)
    
    def load_from_json(self):
        """
        Charge les données depuis un fichier JSON.
        """
        with open(self.filename, 'r') as f:
            self.data = json.load(f)

    def clear_data(self):
        """
        Réinitialise les données stockées.
        """
        self.data = {
            'trajectory': [],
            'parameters': {}
        }