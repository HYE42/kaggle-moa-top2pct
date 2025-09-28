# %% [code] {"_kg_hide-input":true,"execution":{"iopub.execute_input":"2020-10-30T16:26:44.160903Z","iopub.status.busy":"2020-10-30T16:26:44.160036Z","iopub.status.idle":"2020-10-30T16:26:45.599387Z","shell.execute_reply":"2020-10-30T16:26:45.598462Z"},"executionInfo":{"elapsed":6322,"status":"ok","timestamp":1605114236516,"user":{"displayName":"Vladimir Zhuravlev","photoUrl":"","userId":"16372324542816680996"},"user_tz":-420},"id":"pfi5SZ_kU50f","outputId":"adaffd70-a7f7-49e6-ee61-20961c80c5d4","papermill":{"duration":1.50169,"end_time":"2020-10-30T16:26:45.599507","exception":false,"start_time":"2020-10-30T16:26:44.097817","status":"completed"},"tags":[]}
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import seaborn as sns
 
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
 
import warnings
warnings.filterwarnings('ignore')

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.execute_input":"2020-10-30T16:26:45.771303Z","iopub.status.busy":"2020-10-30T16:26:45.770451Z","iopub.status.idle":"2020-10-30T16:26:45.778663Z","shell.execute_reply":"2020-10-30T16:26:45.779209Z"},"executionInfo":{"elapsed":79185,"status":"ok","timestamp":1605114309438,"user":{"displayName":"Vladimir Zhuravlev","photoUrl":"","userId":"16372324542816680996"},"user_tz":-420},"id":"PcmCrDMHU50n","outputId":"8d65d771-547c-47c1-b374-dbc0f8d67247","papermill":{"duration":0.050111,"end_time":"2020-10-30T16:26:45.779337","exception":false,"start_time":"2020-10-30T16:26:45.729226","status":"completed"},"tags":[]}
data_dir = '../input/lish-moa/'
os.listdir(data_dir)

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.execute_input":"2020-10-30T16:26:45.864219Z","iopub.status.busy":"2020-10-30T16:26:45.863163Z","iopub.status.idle":"2020-10-30T16:26:53.257922Z","shell.execute_reply":"2020-10-30T16:26:53.257217Z"},"id":"m7vLwXCDU50r","papermill":{"duration":7.440368,"end_time":"2020-10-30T16:26:53.258045","exception":false,"start_time":"2020-10-30T16:26:45.817677","status":"completed"},"tags":[]}
train_features = pd.read_csv(data_dir + 'train_features.csv')
train_targets_scored = pd.read_csv(data_dir + 'train_targets_scored.csv')
train_targets_nonscored = pd.read_csv(data_dir + 'train_targets_nonscored.csv')
train_drug = pd.read_csv(data_dir + 'train_drug.csv')
test_features = pd.read_csv(data_dir + 'test_features.csv')
sample_submission = pd.read_csv(data_dir + 'sample_submission.csv')

"""print('train_features: {}'.format(train_features.shape))
print('train_targets_scored: {}'.format(train_targets_scored.shape))
print('train_targets_nonscored: {}'.format(train_targets_nonscored.shape))
print('train_drug: {}'.format(train_drug.shape))
print('test_features: {}'.format(test_features.shape))
print('sample_submission: {}'.format(sample_submission.shape))"""

train_features2=train_features.copy() #
test_features2=test_features.copy() #

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T16:26:53.34316Z","iopub.status.busy":"2020-10-30T16:26:53.342367Z","iopub.status.idle":"2020-10-30T16:26:53.346789Z","shell.execute_reply":"2020-10-30T16:26:53.346264Z"},"id":"YbP28luJU50t","papermill":{"duration":0.048987,"end_time":"2020-10-30T16:26:53.346888","exception":false,"start_time":"2020-10-30T16:26:53.297901","status":"completed"},"tags":[]}
GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]

#print('GENES: {}'.format(GENES[:10]))
#print('CELLS: {}'.format(CELLS[:10]))

# %% [code]
#!pip install /kaggle/input/iterative-stratification/iterative-stratification-master/
#from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# %% [markdown] {"id":"-gsULXIEVs1V"}
# # RankGauss

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T16:27:02.692315Z","iopub.status.busy":"2020-10-30T16:27:02.691473Z","iopub.status.idle":"2020-10-30T16:27:12.807731Z","shell.execute_reply":"2020-10-30T16:27:12.807143Z"},"id":"XmZk8BSxU509","papermill":{"duration":10.171313,"end_time":"2020-10-30T16:27:12.807963","exception":false,"start_time":"2020-10-30T16:27:02.63665","status":"completed"},"tags":[]}
import pickle

i = 0
for col in (GENES + CELLS):
    transformer = QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal")
    vec_len = len(train_features[col].values)
    vec_len_test = len(test_features[col].values)
    raw_vec = train_features[col].values.reshape(vec_len, 1)
    
    transformer = pickle.load(open(f'../input/13-kf/tm_{i}.pkl', 'rb'))  ##1
    i += 1
    #transformer.fit(raw_vec)

    train_features[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
    test_features[col] = transformer.transform(test_features[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T16:27:12.990981Z","iopub.status.busy":"2020-10-30T16:27:12.990145Z","iopub.status.idle":"2020-10-30T16:27:12.996237Z","shell.execute_reply":"2020-10-30T16:27:12.995593Z"},"id":"6J2XxF2lU51A","papermill":{"duration":0.057375,"end_time":"2020-10-30T16:27:12.996346","exception":false,"start_time":"2020-10-30T16:27:12.938971","status":"completed"},"tags":[]}
SEED_VALUE = 42

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=SEED_VALUE)

# %% [markdown] {"id":"zqqSsd-0U51O","papermill":{"duration":0.071318,"end_time":"2020-10-30T16:27:21.173019","exception":false,"start_time":"2020-10-30T16:27:21.101701","status":"completed"},"tags":[]}
# # PCA features + Existing features

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T16:27:21.327068Z","iopub.status.busy":"2020-10-30T16:27:21.32619Z","iopub.status.idle":"2020-10-30T16:27:34.313177Z","shell.execute_reply":"2020-10-30T16:27:34.312497Z"},"id":"_7XdDzbRU51P","papermill":{"duration":13.068568,"end_time":"2020-10-30T16:27:34.313298","exception":false,"start_time":"2020-10-30T16:27:21.24473","status":"completed"},"tags":[]}
# GENES
n_comp = 600

data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])

pca_g = PCA(n_components=n_comp, random_state=SEED_VALUE)
pca_g = pickle.load(open('../input/13-kf/pca_g.pkl', 'rb')) ##2
data2 = pca_g.transform(data[GENES])

#data2 = (PCA(n_components=n_comp, random_state=SEED_VALUE).fit_transform(data[GENES]))

train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])
test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])

# drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]
train_features = pd.concat((train_features, train2), axis=1)
test_features = pd.concat((test_features, test2), axis=1)

print('train_features: {}'.format(train_features.shape))
print('test_features: {}'.format(test_features.shape))

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T16:27:34.426606Z","iopub.status.busy":"2020-10-30T16:27:34.425198Z","iopub.status.idle":"2020-10-30T16:27:35.183509Z","shell.execute_reply":"2020-10-30T16:27:35.184065Z"},"id":"0iT3PxLmU51R","papermill":{"duration":0.820845,"end_time":"2020-10-30T16:27:35.184223","exception":false,"start_time":"2020-10-30T16:27:34.363378","status":"completed"},"tags":[]}
# CELLS
n_comp = 50

data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])

pca_c = PCA(n_components=n_comp, random_state=SEED_VALUE)
pca_c = pickle.load(open('../input/13-kf/pca_c.pkl', 'rb')) ##3
data2 = pca_c.transform(data[CELLS])


#data2 = (PCA(n_components=n_comp, random_state=SEED_VALUE).fit_transform(data[CELLS]))
train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])
test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])

# drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]
train_features = pd.concat((train_features, train2), axis=1)
test_features = pd.concat((test_features, test2), axis=1)

print('train_features: {}'.format(train_features.shape))
print('test_features: {}'.format(test_features.shape))

# %% [markdown] {"id":"00hCtGDsU51W","papermill":{"duration":0.051098,"end_time":"2020-10-30T16:27:35.401617","exception":false,"start_time":"2020-10-30T16:27:35.350519","status":"completed"},"tags":[]}
# # feature Selection using Variance Encoding

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T16:27:35.517783Z","iopub.status.busy":"2020-10-30T16:27:35.51697Z","iopub.status.idle":"2020-10-30T16:27:36.626838Z","shell.execute_reply":"2020-10-30T16:27:36.625742Z"},"executionInfo":{"elapsed":121315,"status":"ok","timestamp":1605114351712,"user":{"displayName":"Vladimir Zhuravlev","photoUrl":"","userId":"16372324542816680996"},"user_tz":-420},"id":"ri4YPSgxU51X","outputId":"7a1f41b1-5c9f-4f2f-af02-d98dc5f63823","papermill":{"duration":1.173308,"end_time":"2020-10-30T16:27:36.626956","exception":false,"start_time":"2020-10-30T16:27:35.453648","status":"completed"},"tags":[]}
from sklearn.feature_selection import VarianceThreshold

var_thresh = VarianceThreshold(0.8)
data = train_features.append(test_features)

var_thresh = pickle.load(open('../input/13-kf/var_tm.pkl', 'rb'))  ##4
data_transformed = var_thresh.transform(data.iloc[:, 4:])


#data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])

train_features_transformed = data_transformed[ : train_features.shape[0]]
test_features_transformed = data_transformed[-test_features.shape[0] : ]

train_features = pd.DataFrame(train_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                              columns=['sig_id','cp_type','cp_time','cp_dose'])

train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)

test_features = pd.DataFrame(test_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                             columns=['sig_id','cp_type','cp_time','cp_dose'])

test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)

print('train_features: {}'.format(train_features.shape))
print('test_features: {}'.format(test_features.shape))

gsquarecols=['g-574','g-211','g-216','g-0','g-255','g-577','g-153','g-389','g-60','g-370','g-248','g-167','g-203','g-177','g-301','g-332','g-517','g-6','g-744','g-224','g-162','g-3','g-736','g-486','g-283','g-22','g-359','g-361','g-440','g-335','g-106','g-307','g-745','g-146','g-416','g-298','g-666','g-91','g-17','g-549','g-145','g-157','g-768','g-568','g-396']

def fe_stats(train, test):
    
    features_g = GENES
    features_c = CELLS
    
    for df in train, test:
        df['g_sum'] = df[features_g].sum(axis = 1)
        df['g_mean'] = df[features_g].mean(axis = 1)
        df['g_std'] = df[features_g].std(axis = 1)
        df['g_kurt'] = df[features_g].kurtosis(axis = 1)
        df['g_skew'] = df[features_g].skew(axis = 1)
        df['c_sum'] = df[features_c].sum(axis = 1)
        df['c_mean'] = df[features_c].mean(axis = 1)
        df['c_std'] = df[features_c].std(axis = 1)
        df['c_kurt'] = df[features_c].kurtosis(axis = 1)
        df['c_skew'] = df[features_c].skew(axis = 1)
        df['gc_sum'] = df[features_g + features_c].sum(axis = 1)
        df['gc_mean'] = df[features_g + features_c].mean(axis = 1)
        df['gc_std'] = df[features_g + features_c].std(axis = 1)
        df['gc_kurt'] = df[features_g + features_c].kurtosis(axis = 1)
        df['gc_skew'] = df[features_g + features_c].skew(axis = 1)
        
        df['c52_c42'] = df['c-52'] * df['c-42']
        df['c13_c73'] = df['c-13'] * df['c-73']
        df['c26_c13'] = df['c-26'] * df['c-13']
        df['c33_c6'] = df['c-33'] * df['c-6']
        df['c11_c55'] = df['c-11'] * df['c-55']
        df['c38_c63'] = df['c-38'] * df['c-63']
        df['c38_c94'] = df['c-38'] * df['c-94']
        df['c13_c94'] = df['c-13'] * df['c-94']
        df['c4_c52'] = df['c-4'] * df['c-52']
        df['c4_c42'] = df['c-4'] * df['c-42']
        df['c13_c38'] = df['c-13'] * df['c-38']
        df['c55_c2'] = df['c-55'] * df['c-2']
        df['c55_c4'] = df['c-55'] * df['c-4']
        df['c4_c13'] = df['c-4'] * df['c-13']
        df['c82_c42'] = df['c-82'] * df['c-42']
        df['c66_c42'] = df['c-66'] * df['c-42']
        df['c6_c38'] = df['c-6'] * df['c-38']
        df['c2_c13'] = df['c-2'] * df['c-13']
        df['c62_c42'] = df['c-62'] * df['c-42']
        df['c90_c55'] = df['c-90'] * df['c-55']
        
        
        for feature in features_c:
             df[f'{feature}_squared'] = df[feature] ** 2     
                
        for feature in gsquarecols:
            df[f'{feature}_squared'] = df[feature] ** 2        
        
    return train, test

train_features2,test_features2=fe_stats(train_features2,test_features2) #

train_features_stats=train_features2.iloc[:,876:]
test_features_stats=test_features2.iloc[:,876:]

train_features = pd.concat((train_features, train_features_stats), axis=1)
test_features = pd.concat((test_features, test_features_stats), axis=1)

# %% [code]
#train_drug.head()


# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T16:27:36.73753Z","iopub.status.busy":"2020-10-30T16:27:36.736157Z","iopub.status.idle":"2020-10-30T16:27:37.084383Z","shell.execute_reply":"2020-10-30T16:27:37.084913Z"},"id":"0YYLua3IU51a","papermill":{"duration":0.408004,"end_time":"2020-10-30T16:27:37.085091","exception":false,"start_time":"2020-10-30T16:27:36.677087","status":"completed"},"tags":[]}
train = train_features.merge(train_targets_scored, on='sig_id')
train = train.merge(train_targets_nonscored, on='sig_id')
train = train.merge(train_drug, on='sig_id')
train = train[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
test = test_features[test_features['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T16:27:37.275367Z","iopub.status.busy":"2020-10-30T16:27:37.261427Z","iopub.status.idle":"2020-10-30T16:27:37.278339Z","shell.execute_reply":"2020-10-30T16:27:37.277776Z"},"id":"rBQk0po9U51c","papermill":{"duration":0.142461,"end_time":"2020-10-30T16:27:37.278454","exception":false,"start_time":"2020-10-30T16:27:37.135993","status":"completed"},"tags":[]}
train = train.drop('cp_type', axis=1)
test = test.drop('cp_type', axis=1)

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T16:27:37.560176Z","iopub.status.busy":"2020-10-30T16:27:37.558799Z","iopub.status.idle":"2020-10-30T16:27:37.562449Z","shell.execute_reply":"2020-10-30T16:27:37.561877Z"},"executionInfo":{"elapsed":122330,"status":"ok","timestamp":1605114352785,"user":{"displayName":"Vladimir Zhuravlev","photoUrl":"","userId":"16372324542816680996"},"user_tz":-420},"id":"S4hXidfQU51h","outputId":"f1b594f9-0c92-41c1-f88d-cfe75628b7ef","papermill":{"duration":0.074948,"end_time":"2020-10-30T16:27:37.562559","exception":false,"start_time":"2020-10-30T16:27:37.487611","status":"completed"},"tags":[]}
target_cols = [x for x in train_targets_scored.columns if x != 'sig_id']
aux_target_cols = [x for x in train_targets_nonscored.columns if x != 'sig_id']
all_target_cols = target_cols + aux_target_cols

num_targets = len(target_cols)
"""num_aux_targets = len(aux_target_cols)
num_all_targets = len(all_target_cols)

print('num_targets: {}'.format(num_targets))
print('num_aux_targets: {}'.format(num_aux_targets))
print('num_all_targets: {}'.format(num_all_targets))"""

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T16:27:41.055125Z","iopub.status.busy":"2020-10-30T16:27:41.054094Z","iopub.status.idle":"2020-10-30T16:27:41.059748Z","shell.execute_reply":"2020-10-30T16:27:41.060478Z"},"executionInfo":{"elapsed":125156,"status":"ok","timestamp":1605114355642,"user":{"displayName":"Vladimir Zhuravlev","photoUrl":"","userId":"16372324542816680996"},"user_tz":-420},"id":"xikvp4WLU51m","outputId":"96a60f5d-6af6-4f99-e753-17f63406623d","papermill":{"duration":0.065246,"end_time":"2020-10-30T16:27:41.060624","exception":false,"start_time":"2020-10-30T16:27:40.995378","status":"completed"},"tags":[]}
#print(train.shape)
#print(test.shape)
#print(sample_submission.shape)

# %% [markdown] {"id":"-4PxBYMVU51p","papermill":{"duration":0.053045,"end_time":"2020-10-30T16:27:41.168745","exception":false,"start_time":"2020-10-30T16:27:41.1157","status":"completed"},"tags":[]}
# # Dataset Classes

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T16:27:41.290886Z","iopub.status.busy":"2020-10-30T16:27:41.289986Z","iopub.status.idle":"2020-10-30T16:27:41.293444Z","shell.execute_reply":"2020-10-30T16:27:41.292871Z"},"id":"Oe4Fyj0rU51q","papermill":{"duration":0.069081,"end_time":"2020-10-30T16:27:41.293559","exception":false,"start_time":"2020-10-30T16:27:41.224478","status":"completed"},"tags":[]}
class MoADataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)
        }
        
        return dct
    
class TestDataset:
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }

        return dct

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T16:27:41.419307Z","iopub.status.busy":"2020-10-30T16:27:41.415376Z","iopub.status.idle":"2020-10-30T16:27:41.421889Z","shell.execute_reply":"2020-10-30T16:27:41.422529Z"},"id":"tIvXQTmIU51s","papermill":{"duration":0.074766,"end_time":"2020-10-30T16:27:41.422664","exception":false,"start_time":"2020-10-30T16:27:41.347898","status":"completed"},"tags":[]}
def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0
    
    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        final_loss += loss.item()
        
    final_loss /= len(dataloader)
    return final_loss

def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []
    
    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)
    return final_loss, valid_preds

def inference_fn(model, dataloader, device):  #.inf
    model.eval()
    preds = []
    
    for data in dataloader:
        inputs = data['x'].to(device)

        with torch.no_grad():
            outputs = model(inputs)
        
        preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    preds = np.concatenate(preds)
    return preds

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T16:27:41.544517Z","iopub.status.busy":"2020-10-30T16:27:41.543653Z","iopub.status.idle":"2020-10-30T16:27:41.546741Z","shell.execute_reply":"2020-10-30T16:27:41.546261Z"},"id":"xEI0c6n2U51u","papermill":{"duration":0.069859,"end_time":"2020-10-30T16:27:41.546868","exception":false,"start_time":"2020-10-30T16:27:41.477009","status":"completed"},"tags":[]}
import torch
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1

        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
            
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

# %% [markdown] {"id":"2YOmMelYU51x","papermill":{"duration":0.055406,"end_time":"2020-10-30T16:27:41.65766","exception":false,"start_time":"2020-10-30T16:27:41.602254","status":"completed"},"tags":[]}
# # Model

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T16:27:41.785992Z","iopub.status.busy":"2020-10-30T16:27:41.784898Z","iopub.status.idle":"2020-10-30T16:27:41.787531Z","shell.execute_reply":"2020-10-30T16:27:41.788132Z"},"id":"NuBU5W3lU51y","papermill":{"duration":0.076376,"end_time":"2020-10-30T16:27:41.78827","exception":false,"start_time":"2020-10-30T16:27:41.711894","status":"completed"},"tags":[]}
class Model(nn.Module):
    def __init__(self, num_features, num_targets):
        super(Model, self).__init__()
        self.hidden_size = [1500, 1250, 1000, 750]
        self.dropout_value = [0.5, 0.35, 0.3, 0.25]

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.Linear(num_features, self.hidden_size[0])
        
        self.batch_norm2 = nn.BatchNorm1d(self.hidden_size[0])
        self.dropout2 = nn.Dropout(self.dropout_value[0])
        self.dense2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])

        self.batch_norm3 = nn.BatchNorm1d(self.hidden_size[1])
        self.dropout3 = nn.Dropout(self.dropout_value[1])
        self.dense3 = nn.Linear(self.hidden_size[1], self.hidden_size[2])

        self.batch_norm4 = nn.BatchNorm1d(self.hidden_size[2])
        self.dropout4 = nn.Dropout(self.dropout_value[2])
        self.dense4 = nn.Linear(self.hidden_size[2], self.hidden_size[3])

        self.batch_norm5 = nn.BatchNorm1d(self.hidden_size[3])
        self.dropout5 = nn.Dropout(self.dropout_value[3])
        self.dense5 = nn.utils.weight_norm(nn.Linear(self.hidden_size[3], num_targets))
    
    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = F.leaky_relu(self.dense3(x))

        x = self.batch_norm4(x)
        x = self.dropout4(x)
        x = F.leaky_relu(self.dense4(x))

        x = self.batch_norm5(x)
        x = self.dropout5(x)
        x = self.dense5(x)
        return x
    
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# %% [code] {"id":"NJaTGiY4uJRP"}
class FineTuneScheduler:
    def __init__(self, epochs):
        self.epochs = epochs
        self.epochs_per_step = 0
        self.frozen_layers = []

    def copy_without_top(self, model, num_features, num_targets, num_targets_new):
        self.frozen_layers = []

        model_new = Model(num_features, num_targets)
        model_new.load_state_dict(model.state_dict())

        # Freeze all weights
        for name, param in model_new.named_parameters():
            layer_index = name.split('.')[0][-1]

            if layer_index == 5:
                continue

            param.requires_grad = False

            # Save frozen layer names
            if layer_index not in self.frozen_layers:
                self.frozen_layers.append(layer_index)

        self.epochs_per_step = self.epochs // len(self.frozen_layers)

        # Replace the top layers with another ones
        model_new.batch_norm5 = nn.BatchNorm1d(model_new.hidden_size[3])
        model_new.dropout5 = nn.Dropout(model_new.dropout_value[3])
        model_new.dense5 = nn.utils.weight_norm(nn.Linear(model_new.hidden_size[-1], num_targets_new))
        model_new.to(DEVICE)
        return model_new

    def step(self, epoch, model):
        if len(self.frozen_layers) == 0:
            return

        if epoch % self.epochs_per_step == 0:
            last_frozen_index = self.frozen_layers[-1]
            
            # Unfreeze parameters of the last frozen layer
            for name, param in model.named_parameters():
                layer_index = name.split('.')[0][-1]

                if layer_index == last_frozen_index:
                    param.requires_grad = True

            del self.frozen_layers[-1]  # Remove the last layer as unfrozen

# %% [markdown] {"id":"5XUc9N9zU512","papermill":{"duration":0.054326,"end_time":"2020-10-30T16:27:41.896645","exception":false,"start_time":"2020-10-30T16:27:41.842319","status":"completed"},"tags":[]}
# # Preprocessing steps

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T16:27:42.014179Z","iopub.status.busy":"2020-10-30T16:27:42.012925Z","iopub.status.idle":"2020-10-30T16:27:42.014885Z","shell.execute_reply":"2020-10-30T16:27:42.01538Z"},"id":"PUxdopjmU513","papermill":{"duration":0.063622,"end_time":"2020-10-30T16:27:42.015502","exception":false,"start_time":"2020-10-30T16:27:41.95188","status":"completed"},"tags":[]}
def process_data(data):
    data = pd.get_dummies(data, columns=['cp_time','cp_dose'])
    return data

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T16:27:42.134029Z","iopub.status.busy":"2020-10-30T16:27:42.13261Z","iopub.status.idle":"2020-10-30T16:27:42.325649Z","shell.execute_reply":"2020-10-30T16:27:42.326231Z"},"executionInfo":{"elapsed":126139,"status":"ok","timestamp":1605114356736,"user":{"displayName":"Vladimir Zhuravlev","photoUrl":"","userId":"16372324542816680996"},"user_tz":-420},"id":"lcFRstDpU515","outputId":"68d18d46-efda-46d4-bfe4-d641bcbc5fe8","papermill":{"duration":0.256088,"end_time":"2020-10-30T16:27:42.32637","exception":false,"start_time":"2020-10-30T16:27:42.070282","status":"completed"},"tags":[]}
feature_cols = [c for c in process_data(train).columns if c not in all_target_cols]
feature_cols = [c for c in feature_cols if c not in ['kfold', 'sig_id', 'drug_id']]
num_features = len(feature_cols)
num_features

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T16:27:42.973092Z","iopub.status.busy":"2020-10-30T16:27:42.9717Z","iopub.status.idle":"2020-10-30T16:27:42.976302Z","shell.execute_reply":"2020-10-30T16:27:42.978092Z"},"id":"J80ID9aaU518","papermill":{"duration":0.596642,"end_time":"2020-10-30T16:27:42.978321","exception":false,"start_time":"2020-10-30T16:27:42.381679","status":"completed"},"tags":[]}
# HyperParameters

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 24
BATCH_SIZE = 128

WEIGHT_DECAY = {'ALL_TARGETS': 1e-5, 'SCORED_ONLY': 3e-6}
MAX_LR = {'ALL_TARGETS': 1e-2, 'SCORED_ONLY': 3e-3}
DIV_FACTOR = {'ALL_TARGETS': 1e3, 'SCORED_ONLY': 1e2}
PCT_START = 0.1

# %% [code] {"executionInfo":{"elapsed":126118,"status":"ok","timestamp":1605114356745,"user":{"displayName":"Vladimir Zhuravlev","photoUrl":"","userId":"16372324542816680996"},"user_tz":-420},"id":"7d2R5qElopmx","outputId":"747b3d73-b306-48da-f488-9d7e146ce5dd"}
# Show model architecture
#model = Model(num_features, num_all_targets)
#model

# %% [markdown] {"id":"0hidUPH8U51-","papermill":{"duration":0.149218,"end_time":"2020-10-30T16:27:43.27826","exception":false,"start_time":"2020-10-30T16:27:43.129042","status":"completed"},"tags":[]}
# # Single fold training

# %% [code]
from sklearn.model_selection import KFold
"""
def make_cv_folds(train, SEEDS, NFOLDS, DRUG_THRESH):
    vc = train.drug_id.value_counts()
    vc1 = vc.loc[vc <= DRUG_THRESH].index.sort_values()
    vc2 = vc.loc[vc > DRUG_THRESH].index.sort_values()

    for seed_id in range(SEEDS):
        kfold_col = 'kfold_{}'.format(seed_id)
        
        # STRATIFY DRUGS 18X OR LESS
        dct1 = {}
        dct2 = {}

        skf = MultilabelStratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=seed_id)
        tmp = train.groupby('drug_id')[target_cols].mean().loc[vc1]

        for fold,(idxT, idxV) in enumerate(skf.split(tmp, tmp[target_cols])):
            dd = {k: fold for k in tmp.index[idxV].values}
            dct1.update(dd)

        # STRATIFY DRUGS MORE THAN 18X
        skf = MultilabelStratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=seed_id)
        tmp = train.loc[train.drug_id.isin(vc2)].reset_index(drop=True)

        for fold,(idxT, idxV) in enumerate(skf.split(tmp, tmp[target_cols])):
            dd = {k: fold for k in tmp.sig_id[idxV].values}
            dct2.update(dd)

        # ASSIGN FOLDS
        train[kfold_col] = train.drug_id.map(dct1)
        train.loc[train[kfold_col].isna(), kfold_col] = train.loc[train[kfold_col].isna(), 'sig_id'].map(dct2)
        train[kfold_col] = train[kfold_col].astype('int8')
        
    return train
"""
SEEDS = 7
NFOLDS = 7
DRUG_THRESH = 18

#train = make_cv_folds(train, SEEDS, NFOLDS, DRUG_THRESH)
#train.head()

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T16:27:43.533519Z","iopub.status.busy":"2020-10-30T16:27:43.532417Z","iopub.status.idle":"2020-10-30T16:27:43.586422Z","shell.execute_reply":"2020-10-30T16:27:43.585623Z"},"id":"JHTrp23wU51_","papermill":{"duration":0.213528,"end_time":"2020-10-30T16:27:43.586582","exception":false,"start_time":"2020-10-30T16:27:43.373054","status":"completed"},"tags":[]}
def run_training(fold_id, seed_id):
    seed_everything(seed_id)
    
    #train_ = process_data(train)
    test_ = process_data(test)
    
    kfold_col = f'kfold_{seed_id}'
    #trn_idx = train_[train_[kfold_col] != fold_id].index
    #val_idx = train_[train_[kfold_col] == fold_id].index
    
    #train_df = train_[train_[kfold_col] != fold_id].reset_index(drop=True)
    #valid_df = train_[train_[kfold_col] == fold_id].reset_index(drop=True)
    """
    def train_model(model, tag_name, target_cols_now, fine_tune_scheduler=None):
        x_train, y_train  = train_df[feature_cols].values, train_df[target_cols_now].values
        x_valid, y_valid =  valid_df[feature_cols].values, valid_df[target_cols_now].values
        
        train_dataset = MoADataset(x_train, y_train)
        valid_dataset = MoADataset(x_valid, y_valid)

        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY[tag_name])
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                  steps_per_epoch=len(trainloader),
                                                  pct_start=PCT_START,
                                                  div_factor=DIV_FACTOR[tag_name], 
                                                  max_lr=MAX_LR[tag_name],
                                                  epochs=EPOCHS)
        
        loss_fn = nn.BCEWithLogitsLoss()
        loss_tr = SmoothBCEwLogits(smoothing=0.001)

        oof = np.zeros((len(train), len(target_cols_now)))
        best_loss = np.inf
        
        for epoch in range(EPOCHS):
            if fine_tune_scheduler is not None:
                fine_tune_scheduler.step(epoch, model)

            train_loss = train_fn(model, optimizer, scheduler, loss_tr, trainloader, DEVICE)
            valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)
            print(f"SEED: {seed_id}, FOLD: {fold_id}, {tag_name}, EPOCH: {epoch}, train_loss: {train_loss:.6f}, valid_loss: {valid_loss:.6f}")

            if np.isnan(valid_loss):
                break
            
            if valid_loss < best_loss:
                best_loss = valid_loss
                oof[val_idx] = valid_preds
                torch.save(model.state_dict(), f"{tag_name}_FOLD{fold_id}_SEED{seed_id}.pth")

        return oof

    fine_tune_scheduler = FineTuneScheduler(EPOCHS)

    pretrained_model = Model(num_features, num_all_targets)
    pretrained_model.to(DEVICE)

    # Train on scored + nonscored targets
    train_model(pretrained_model, 'ALL_TARGETS', all_target_cols) 

    # Load the pretrained model with the best loss
    pretrained_model = Model(num_features, num_all_targets)
    pretrained_model.load_state_dict(torch.load(f"ALL_TARGETS_FOLD{fold_id}_SEED{seed_id}.pth"))
    pretrained_model.to(DEVICE)

    # Copy model without the top layer
    final_model = fine_tune_scheduler.copy_without_top(pretrained_model, num_features, num_all_targets, num_targets)

    # Fine-tune the model on scored targets only
    oof = train_model(final_model, 'SCORED_ONLY', target_cols, fine_tune_scheduler)
    """
    #--------------------- above: train---------------------

    # Load the fine-tuned model with the best loss
    model = Model(num_features, num_targets)            ########↓
    model.load_state_dict(torch.load(f"../input/13-kf/SCORED_ONLY_FOLD{fold_id}_SEED{seed_id}.pth")) ##5
    model.to(DEVICE)

    #--------------------- PREDICTION---------------------
    x_test = test_[feature_cols].values  ##
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
    
    predictions = np.zeros((len(test_), num_targets))
    predictions = inference_fn(model, testloader, DEVICE)
    return predictions #oof,

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T16:27:43.804473Z","iopub.status.busy":"2020-10-30T16:27:43.803598Z","iopub.status.idle":"2020-10-30T16:27:43.808545Z","shell.execute_reply":"2020-10-30T16:27:43.809223Z"},"id":"GxfyrnG4U52C","papermill":{"duration":0.143086,"end_time":"2020-10-30T16:27:43.809403","exception":false,"start_time":"2020-10-30T16:27:43.666317","status":"completed"},"tags":[]}
def run_k_fold(NFOLDS, seed_id):
    #oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))
    
    for fold_id in range(NFOLDS):
        pred_ = run_training(fold_id, seed_id) #oof_, 
        predictions += pred_ / NFOLDS
        #oof += oof_
        
    return predictions #oof,

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T16:27:43.965401Z","iopub.status.busy":"2020-10-30T16:27:43.964403Z","iopub.status.idle":"2020-10-30T17:02:14.476596Z","shell.execute_reply":"2020-10-30T17:02:14.475472Z"},"executionInfo":{"elapsed":4373947,"status":"ok","timestamp":1605118604618,"user":{"displayName":"Vladimir Zhuravlev","photoUrl":"","userId":"16372324542816680996"},"user_tz":-420},"id":"Tix4uSLRU52D","outputId":"7fba4206-ad29-4c55-960a-14abc3ecebdd","papermill":{"duration":2070.583824,"end_time":"2020-10-30T17:02:14.476733","exception":false,"start_time":"2020-10-30T16:27:43.892909","status":"completed"},"tags":[]}
#from time import time

# Averaging on multiple SEEDS
SEED = [0, 1, 2, 3, 4, 5, 6]
#oof = np.zeros((len(train), len(target_cols)))
predictions = np.zeros((len(test), len(target_cols)))

#time_begin = time()

for seed_id in SEED:
    predictions_ = run_k_fold(NFOLDS, seed_id) #oof_, 
    #oof += oof_ / len(SEED)
    predictions += predictions_ / len(SEED)

#time_diff = time() - time_begin

#train[target_cols] = oof
test[target_cols] = predictions

# %% [code] {"executionInfo":{"elapsed":4374004,"status":"ok","timestamp":1605118604690,"user":{"displayName":"Vladimir Zhuravlev","photoUrl":"","userId":"16372324542816680996"},"user_tz":-420},"id":"7ByqrKu0LFhe","outputId":"b42229e5-3187-435c-f10e-5a7a0b74f7bf"}
"""from datetime import timedelta
str(timedelta(seconds=time_diff))"""

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T17:02:15.499323Z","iopub.status.busy":"2020-10-30T17:02:15.49831Z","iopub.status.idle":"2020-10-30T17:02:15.526114Z","shell.execute_reply":"2020-10-30T17:02:15.526649Z"},"executionInfo":{"elapsed":4373987,"status":"ok","timestamp":1605118604695,"user":{"displayName":"Vladimir Zhuravlev","photoUrl":"","userId":"16372324542816680996"},"user_tz":-420},"id":"venT86yMU52F","outputId":"c6b013a7-b986-4677-e12b-6a9e965f69fb","papermill":{"duration":0.549441,"end_time":"2020-10-30T17:02:15.526794","exception":false,"start_time":"2020-10-30T17:02:14.977353","status":"completed"},"tags":[]}
#train_targets_scored.head()

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T17:02:16.601107Z","iopub.status.busy":"2020-10-30T17:02:16.600285Z","iopub.status.idle":"2020-10-30T17:02:16.604255Z","shell.execute_reply":"2020-10-30T17:02:16.604744Z"},"executionInfo":{"elapsed":4373977,"status":"ok","timestamp":1605118604700,"user":{"displayName":"Vladimir Zhuravlev","photoUrl":"","userId":"16372324542816680996"},"user_tz":-420},"id":"lSATXXgZU52H","outputId":"9092dbc1-8c0b-4669-db3e-5f22fb92f860","papermill":{"duration":0.53223,"end_time":"2020-10-30T17:02:16.604888","exception":false,"start_time":"2020-10-30T17:02:16.072658","status":"completed"},"tags":[]}
#len(target_cols)

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T17:02:17.645921Z","iopub.status.busy":"2020-10-30T17:02:17.644851Z","iopub.status.idle":"2020-10-30T17:02:19.277871Z","shell.execute_reply":"2020-10-30T17:02:19.27717Z"},"executionInfo":{"elapsed":4375017,"status":"ok","timestamp":1605118605755,"user":{"displayName":"Vladimir Zhuravlev","photoUrl":"","userId":"16372324542816680996"},"user_tz":-420},"id":"j1s9_62IU52L","outputId":"afd34d19-0e09-4f4f-efeb-427648cc0f51","papermill":{"duration":2.164653,"end_time":"2020-10-30T17:02:19.277983","exception":false,"start_time":"2020-10-30T17:02:17.11333","status":"completed"},"tags":[]}
"""valid_results = train_targets_scored.drop(columns=target_cols).merge(train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

y_true = train_targets_scored[target_cols].values
y_pred = valid_results[target_cols].values

score = 0

for i in range(len(target_cols)):
    score += log_loss(y_true[:, i], y_pred[:, i])

print("CV log_loss: ", score / y_pred.shape[1])"""

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T17:02:21.416721Z","iopub.status.busy":"2020-10-30T17:02:21.415189Z","iopub.status.idle":"2020-10-30T17:02:23.900023Z","shell.execute_reply":"2020-10-30T17:02:23.898834Z"},"id":"Wl2DgUjxU52R","papermill":{"duration":3.028268,"end_time":"2020-10-30T17:02:23.900162","exception":false,"start_time":"2020-10-30T17:02:20.871894","status":"completed"},"tags":[]}
sub = sample_submission.drop(columns=target_cols).merge(test[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)
sub.to_csv('submission1.csv', index=False)

# %% [code] {"execution":{"iopub.execute_input":"2020-10-30T17:02:24.92982Z","iopub.status.busy":"2020-10-30T17:02:24.92886Z","iopub.status.idle":"2020-10-30T17:02:24.932616Z","shell.execute_reply":"2020-10-30T17:02:24.933157Z"},"executionInfo":{"elapsed":4377655,"status":"ok","timestamp":1605118608442,"user":{"displayName":"Vladimir Zhuravlev","photoUrl":"","userId":"16372324542816680996"},"user_tz":-420},"id":"YPsx4R36U52T","outputId":"1f5dba12-b2dc-4d51-adf5-c7029ffa85e9","papermill":{"duration":0.521397,"end_time":"2020-10-30T17:02:24.933279","exception":false,"start_time":"2020-10-30T17:02:24.411882","status":"completed"},"tags":[]}
#sub.shape