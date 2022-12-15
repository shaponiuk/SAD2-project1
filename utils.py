import scanpy as sc
import os
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

def get_adata():
    train_data = sc.read_h5ad('data/SAD2022Z_Project1_GEX_train.h5ad')
    test_data = sc.read_h5ad('data/SAD2022Z_Project1_GEX_test.h5ad')

    return train_data, test_data

def get_adata_torch(raw):
    train_data, test_data = get_adata()

    if raw:
        train_ = train_data.layers['counts'].toarray()
        test_ = test_data.layers['counts'].toarray()
    else:
        train_ = train_data.X.toarray()
        test_ = test_data.X.toarray()
        
    data_train_t = torch.tensor(train_, dtype=torch.float32)
    data_test_t = torch.tensor(test_, dtype=torch.float32)

    return data_train_t, data_test_t

def get_cell_type_label_encoder():
    train_data, _ = get_adata()
    le = LabelEncoder()
    le.fit(train_data.obs['cell_type'])
    return le

def get_donor_id_label_encoder():
    train_data, _ = get_adata()
    le = LabelEncoder()
    le.fit(train_data.obs['DonorID'])
    return le

def get_site_label_encoder():
    train_data, _ = get_adata()
    le = LabelEncoder()
    le.fit(train_data.obs['Site'])
    return le

def get_cell_type():
    train_data, test_data = get_adata()
    return train_data.obs['cell_type'], test_data.obs['cell_type']

def get_donor_id():
    train_data, test_data = get_adata()
    return train_data.obs['DonorID'], test_data.obs['DonorID']

def get_site():
    train_data, test_data = get_adata()
    return train_data.obs['Site'], test_data.obs['Site']


def get_adata_datasets(raw, normalise=True):
    train_t, test_t = get_adata_torch(raw)

    cell_type_encoder = get_cell_type_label_encoder()
    donor_id_encoder = get_donor_id_label_encoder()
    site_encoder = get_site_label_encoder()

    mean = 0.
    std = 0.

    train_cell_type, test_cell_type = get_cell_type()
    train_donor_id, test_donor_id = get_donor_id()
    train_site, test_site = get_site()

    train_cell_type_t = torch.tensor(cell_type_encoder.transform(train_cell_type))
    test_cell_type_t = torch.tensor(cell_type_encoder.transform(test_cell_type))
    train_donor_id_t = torch.tensor(donor_id_encoder.transform(train_donor_id))
    test_donor_id_t = torch.tensor(donor_id_encoder.transform(test_donor_id))
    train_site_t = torch.tensor(site_encoder.transform(train_site))
    test_site_t = torch.tensor(site_encoder.transform(test_site))

    train_dataset = RNADataset(train_t, train_cell_type_t, train_donor_id_t, train_site_t, mean, std)
    test_dataset = RNADataset(test_t, test_cell_type_t, test_donor_id_t, test_site_t, mean, std)
    
    return train_dataset, test_dataset

def get_dense_x_and_raw(adata):
    x = adata.X.toarray()
    x = x.reshape(-1)
    raw = adata.layers['counts'].toarray()
    raw = raw.reshape(-1)
    
    return x, raw

def preprocess_10k(adata):
    return sc.pp.normalize_total(adata, target_sum=1e4, layer='tmp')

def create_out_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class RNADataset(Dataset):
    def __init__(self, x_data, cell_type, donor_id, site,mean, std):
        self.data = x_data
        self.data -= mean
        self.data /= (std + 1)
        self.cell_type = cell_type
        self.donor_id = donor_id
        self.site = site
        self.mean = mean
        self.std = std

    def dataloader(self, batch_size):
        return DataLoader(self, batch_size)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.cell_type[idx], self.donor_id[idx], self.site[idx]

def kl_divergence(mean, std):
    return (std**2 + mean**2 - torch.log(std) - 1/2).mean()

def sample_normal(loc, scale):
    q = torch.distributions.Normal(loc, scale)
    return q.rsample()

def log_prob_normal(loc, scale, x):
    dist = torch.distributions.Normal(loc, scale)
    return dist.log_prob(x).mean()

def log_prob_neg_binom(n_succ, event_prob, x):
    dist = torch.distributions.NegativeBinomial(n_succ, event_prob)
    return dist.log_prob(x).mean()

def log_prob_poisson(mean, x):
    dist = torch.distributions.Poisson(mean)
    return dist.log_prob(x).mean()

def show_image(path):
    plt.rcParams["figure.figsize"] = (20,20)
    plt.imshow(plt.imread(path))