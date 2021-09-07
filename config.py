class Config:
    MODEL_ROOT = "/content/drive/MyDrive/Colab Notebooks/HumanFacesRecognition/Models_Pytorch_MobilenetV3/"
    LOG_ROOT = 'output/deepcam_log'

    RGB_MEAN = [0.5, 0.5, 0.5]
    RGB_STD = [0.5, 0.5, 0.5]
    BATCH_SIZE = 8
    DROP_LAST = True
    LEARNING_RATE = 0.05
    NUM_EPOCH = 20
    WEIGHT_DECAY = 5e-4
    MOMENTUM = 0.9
    NUM_EPOCH_WARM_UP = 1

config = Config()