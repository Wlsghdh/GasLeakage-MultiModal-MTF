import torch

DATA_DIR_SENSOR = '/data2/project/2025summer/jjh0709/Data/zkwgkjkjn9-2/Gas Sensor HM'
DATA_DIR_SENSOR_CSV ='path = "/data2/project/2025summer/jjh0709/Data/zkwgkjkjn9-2/Gas Sensors Measurements/Gas_Sensors_Measurements.csv'
DATA_DIR_THERMAL = '/data2/project/2025summer/jjh0709/Data/zkwgkjkjn9-2/Thermal Camera Images'

BATCH_SIZE = 8
EPOCHS = 3
NUM_CLASSES = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABEL_MAP = {'Mixture': 0, 'NoGas': 1, 'Perfume': 2, 'Smoke': 3}
ORIGINAL_DATA ="/data2/project/2025summer/jjh0709/Data/zkwgkjkjn9-2/Gas Sensors Measurements/Gas_Sensors_Measurements.csv"
TRAIN_CSV_PATH="/data2/project/2025summer/jjh0709/git/GasLeakage-MultiModal-MTF/data/TRAIN_DATA.csv"
TEST_CSV_PATH="/data2/project/2025summer/jjh0709/git/GasLeakage-MultiModal-MTF/data/TEST_DATA.csv"