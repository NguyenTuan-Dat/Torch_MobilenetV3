class Config:
    MODEL_ROOT = "/content/drive/MyDrive/Colab Notebooks/HumanFacesRecognition/Models_Pytorch_MobilenetV3/"
    LOG_ROOT = 'output/deepcam_log'
    TRAIN_FILES = "/content/TrainingData/GlassesMaskHat_ClassifyDataset/train"
    VALID_FILES = "/content/TrainingData/GlassesMaskHat_ClassifyDataset/test"

    PRETRAINED_MODEL = None

    INPUT_SIZE = (112,112)

    RGB_MEAN = [0.5, 0.5, 0.5]
    RGB_STD = [0.5, 0.5, 0.5]
    BATCH_SIZE = 2048
    DROP_LAST = True
    LEARNING_RATE = 0.0001
    NUM_EPOCH = 30
    WEIGHT_DECAY = 5e-4
    MOMENTUM = 0.9
    NUM_EPOCH_WARM_UP = 1
    DEVICE = [0]

config = Config()