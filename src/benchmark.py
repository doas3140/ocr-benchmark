import cv2

class BenchmarkInterface:
    def __init__(self):
        self.name = self.__class__.__name__

    def chars2strings(self, images):
        return ['' for im in images]
    
    def numbers2strings(self, images):
        return ['' for im in images]

    def horizontal_numbers2strings(self, images):
        return ['' for im in images]

    def vertical_numbers2strings(self, images):
        return ['' for im in images]