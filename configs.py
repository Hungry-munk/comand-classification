class Configs:
    """ 
    A class created purely to store some global variable values needed across all files .
    The class is to be used purely for sound preprocessing, model training (variable storage)
    and batch gen
    """

    def __init__(self):
        self.base_dir = './data/'
        # self.spectrogram_configs = {
        #     'nfft' : 256,
        #     'window' : 256, #should match nfft
        #     'stride' : 128 #should be half of nfft
        # }
        self.spectrogram_configs = {
            'frame_length' : 512, 
            'frame_step': 256
        }
        self.train_batch_size = 32
        self.val_batch_size = 16
        self.test_batch_size = ...
        self.target_rate = 16000 #16khz mono audio
        self.label_encodings = {
            "bed": 0,
            "bird": 1,
            "cat": 2,
            "dog": 3,
            "down": 4,
            "eight": 5,
            "five": 6,
            "four": 7,
            "go": 8,
            "happy": 9,
            "house": 10,
            "left": 11,
            "marvin": 12,
            "nine": 13,
            "no": 14,
            "off": 15,
            "on": 16,
            "one": 17,
            "right": 18,
            "seven": 19,
            "sheila": 20,
            "six": 21,
            "stop": 22,
            "three": 23,
            "tree": 24,
            "two": 25,
            "up": 26,
            "wow": 27,
            "yes": 28,
            "zero": 29
        }
        self.num_classes = len(self.label_encodings)
