import numpy as np
import shap
import pandas as pd
import os
# print(os.getcwd())

class ImageNette():
    def __init__(self):
        self.name="ImageNette"
        self.labels=['Cassette player', 'Garbage truck', 'Tench', 'English springer', 'Church', 'Parachute', 'French horn', 'Chain saw', 'Golf ball', 'Gas pump']
        self.num_classes=len(self.labels)

class Pet():
    def __init__(self):
        self.name="Pet"
        self.labels=['Abyssinian','American Bulldog','American Pit Bull Terrier','Basset Hound','Beagle','Bengal','Birman','Bombay','boxer','British Shorthair','Chihuahua','Egyptian Mau','English Cocker Spaniel','English Setter','German Shorthaired','Great Pyrenees','Havanese','Japanese Chin',
                    'Keeshond','Leonberger','Maine Coon','Miniature Pinscher','Newfoundland','Persian','Pomeranian','Pug','Ragdoll','Russian_Blue','Saint Bernard','Samoyed','Scottish Terrier','Shiba_inu','Siamese','Sphynx','Staffordshire Bull Terrier','Wheaten Terrier','Yorkshire Terrier']
        self.num_classes=len(self.labels)

