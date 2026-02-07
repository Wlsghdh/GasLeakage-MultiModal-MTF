import pandas as pd

from config import ORIGINAL_DATA

def generate_data(train_test_data,original_data):
    TRAIN_TEST_DATA = pd.read_csv(train_test_data)
    train_data = TRAIN_TEST_DATA[TRAIN_TEST_DATA['train_test']=='train']
    test_data = TRAIN_TEST_DATA[TRAIN_TEST_DATA['train_test']=='test']
    original_data = pd.read_csv(original_data)

    NoGas_data = original_data[original_data['Gas']=='NoGas']
    NoGas_train_index = train_data[train_data['Gas']=='NoGas']
    NoGas_test_index = test_data[test_data['Gas']=='NoGas']
    NoGas_train_data = NoGas_data[NoGas_data['Serial Number'].isin(NoGas_train_index['Serial Number'])]
    NoGas_test_data = NoGas_data[NoGas_data['Serial Number'].isin(NoGas_test_index['Serial Number'])]
    


    Perfume_data = original_data[original_data['Gas']=='Perfume']
    Perfume_train_index = train_data[train_data['Gas']=='Perfume']
    Perfume_test_index = test_data[test_data['Gas']=='Perfume']
    Perfume_train_data = Perfume_data[Perfume_data['Serial Number'].isin(Perfume_train_index['Serial Number'])]
    Perfume_test_data = Perfume_data[Perfume_data['Serial Number'].isin(Perfume_test_index['Serial Number'])]

    Smoke_data = original_data[original_data['Gas']=='Smoke']
    Smoke_train_index = train_data[train_data['Gas']=='Smoke']
    Smoke_test_index = test_data[test_data['Gas']=='Smoke']
    Smoke_train_data = Smoke_data[Smoke_data['Serial Number'].isin(Smoke_train_index['Serial Number'])]
    Smoke_test_data = Smoke_data[Smoke_data['Serial Number'].isin(Smoke_test_index['Serial Number'])]

    Mixture_data = original_data[original_data['Gas']=='Mixture']
    Mixture_train_index = train_data[train_data['Gas']=='Mixture']
    Mixture_test_index = test_data[test_data['Gas']=='Mixture']
    Mixture_train_data = Mixture_data[Mixture_data['Serial Number'].isin(Mixture_train_index['Serial Number'])]
    Mixture_test_data = Mixture_data[Mixture_data['Serial Number'].isin(Mixture_test_index['Serial Number'])]
    
    train_data = pd.concat([NoGas_train_data,Perfume_train_data,Smoke_train_data,Mixture_train_data])
    test_data = pd.concat([NoGas_test_data,Perfume_test_data,Smoke_test_data,Mixture_test_data])

    
    return train_data,test_data
