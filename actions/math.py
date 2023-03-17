# Author: Leonardo Rossi Leão
# E-mail: leonardo.leao@cnpem.br

import numpy as np
from scipy.signal import butter, lfilter

class Math():

    @staticmethod
    def polar_to_rectangular(radius: float, theta: float) -> tuple:

        """
            Returns an (x,y) coordinate in rectangular coordinates
        """

        # Convert degree to radians
        x = radius * np.cos(theta * np.pi/180)
        y = radius * np.sin(theta * np.pi/180)

        return (x, y)
    
    @staticmethod
    def get_angle(n_points: int, position: float):

        """
            Divides a circle of arbitrary radius into `n_points` and
            returns the angle of `position`
        """

        return (360/n_points)*(position-1)
    
    @staticmethod
    def filter(data: list, fs: int, lowcut: float = None, highcut: float = None, order: int = 6):

        if None not in [lowcut, highcut]:
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return lfilter(b, a, data)