import utils
import os
import pandas as pd
import matplotlib.pyplot as plt
import model
import torch

def generate_summary_1():
    utils.create_out_dir('./outs')
    train_data, test_data = utils.get_adata()
    summary = [train_data.X.shape, test_data.X.shape]
    summary = pd.DataFrame(summary)
    summary.columns = ["num obs", "num variables"]
    summary.index = ["train", "test"]
    summary.to_csv('./outs/summary_1_1.csv')

def rescaled_capped_hist(data, bins, cap, hist_range, output_path):
    plt.cla()

    plt.rcParams["figure.figsize"] = (20,10)
    
    n, bins, rects = plt.hist(data, bins=bins, label=None, range=hist_range, alpha=0.9)

    nsum = sum(n)
    n = [min(x, cap) * nsum / cap for x in n]

    for h, rect in zip(n, rects):
        rect.set_height(h)
    
    plt.savefig(output_path)

def generate_hists_1():
    utils.create_out_dir('./outs/hists_1')
    train_data, test_data = utils.get_adata()
    x_train, raw_train = utils.get_dense_x_and_raw(train_data)
    x_test, raw_test = utils.get_dense_x_and_raw(test_data)

    rescaled_capped_hist(x_train, bins=50, cap=1e7, hist_range=(0, 20), output_path='./outs/hists_1/hist1.png')
    rescaled_capped_hist(raw_train, bins=50, cap=1e7, hist_range=(0, 20), output_path='./outs/hists_1/hist2.png')
    rescaled_capped_hist(x_test, bins=50, cap=1e7, hist_range=(0, 20), output_path='./outs/hists_1/hist3.png')
    rescaled_capped_hist(raw_test, bins=50, cap=1e7, hist_range=(0, 20), output_path='./outs/hists_1/hist4.png')

def generate_hists_2():
    utils.create_out_dir('./outs/hists_2')
    train_data, test_data = utils.get_adata()
    x_train, raw_train = utils.get_dense_x_and_raw(train_data)
    x_test, raw_test = utils.get_dense_x_and_raw(test_data)

    nonzero_x_train = x_train[x_train != 0]
    nonzero_raw_train = raw_train[raw_train != 0]

    nonzero_x_test = x_test[x_test != 0]
    nonzero_raw_test = raw_test[raw_test != 0]

    rescaled_capped_hist(nonzero_x_train, bins=50, cap=1e8, hist_range=(0, 20), output_path='./outs/hists_2/hist1.png')
    rescaled_capped_hist(nonzero_raw_train, bins=50, cap=1e8, hist_range=(0, 20), output_path='./outs/hists_2/hist2.png')
    rescaled_capped_hist(nonzero_x_test, bins=50, cap=1e8, hist_range=(0, 20), output_path='./outs/hists_2/hist3.png')
    rescaled_capped_hist(nonzero_raw_test, bins=50, cap=1e8, hist_range=(0, 20), output_path='./outs/hists_2/hist4.png')

def plot_metrics(train_metrics, test_metrics, name, output_path,):
    plt.cla()
    plt.subplots(1, 1, figsize=(30,20))
    plt.plot(train_metrics, label='train')
    plt.plot(test_metrics, label='train')
    plt.xlabel('epochs')
    plt.ylabel(name)
    plt.title(name)
    plt.savefig(output_path)
    plt.cla()

def generate_obs_summary():
    utils.create_out_dir('./outs')
    train_data, _ = utils.get_adata()

    n_donors = len(train_data.obs['DonorID'].value_counts())
    n_sites = len(train_data.obs['Site'].value_counts())
    n_cell_types = len(train_data.obs['cell_type'].value_counts())

    df = pd.DataFrame([(n_donors, n_sites, n_cell_types)])
    df.columns = ['n_donors', 'n_sites', 'n_cell_types']
    df.to_csv('./outs/summary_1_6.csv')

def plot_vae_losses_elbo_1(elbo_train, elbo_test, root_path, id):
    utils.create_out_dir(root_path)
    plot_metrics(
        elbo_train, 
        elbo_test, 
        'elbo', 
        '{}/elbo_{}.png'.format(root_path, id)
        )

def plot_vae_losses_reconstruction_1(reconstruction_loss_train, reconstruction_loss_test, root_path, id):
    utils.create_out_dir(root_path)
    plot_metrics(
        reconstruction_loss_train, 
        reconstruction_loss_test, 
        'reconstruction_loss', 
        '{}/reconstruction_loss_{}.png'.format(root_path, id)
        )

def plot_vae_losses_kld_1(kld_train, kld_test, root_path, id):
    utils.create_out_dir(root_path)
    plot_metrics(
        kld_train, 
        kld_test, 
        'kl_divergence', 
        '{}/kl_divergence_{}.png'.format(root_path, id)
        )

def plot_vae(root_path, id):
    epochs = torch.load('{}/model_{}_epochs.pt'.format(root_path, id))
    epochs = {k : [x.item() for x in v] for k, v in epochs.items()}

    plot_vae_losses_elbo_1(epochs['train_elbo'], epochs['test_elbo'], root_path, id=id)
    plot_vae_losses_reconstruction_1(epochs['train_recon_loss'], epochs['test_recon_loss'], root_path, id=id)
    plot_vae_losses_kld_1(epochs['train_kld'], epochs['test_kld'], root_path, id=id)

def fit_pca(z):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=z.shape[1])
    pca.fit(z)
    return pca.explained_variance_ratio_.cumsum()

def generate_pca_table():
    import train

    vae1 = torch.load('outs/vanilla_vae/model_1.pt') 
    vae2 = torch.load('outs/vanilla_vae/model_2.pt') 
    vae3 = torch.load('outs/vanilla_vae/model_3.pt') 

    test_dataloader = utils.get_adata_datasets(raw=True)[1].dataloader(128)

    elbo1, kld1, recon_loss_1, z1, _, _, _, = train.test_a_dataset(vae1, test_dataloader)
    elbo2, kld2, recon_loss_2, z2, _, _, _, = train.test_a_dataset(vae2, test_dataloader)
    elbo3, kld3, recon_loss_3, z3, _, _, _, = train.test_a_dataset(vae3, test_dataloader)

    expl_var_1 = fit_pca(z1)
    expl_var_2 = fit_pca(z2)
    expl_var_3 = fit_pca(z3)
    
    min_components_1 = 1 + len(expl_var_1[expl_var_1 < 0.95])
    min_components_2 = 1 + len(expl_var_2[expl_var_2 < 0.95])
    min_components_3 = 1 + len(expl_var_3[expl_var_3 < 0.95])

    df = pd.DataFrame(
        [
            (z1.shape[1], elbo1.item(), kld1.item(), -recon_loss_1.item(), min_components_1, expl_var_1[min_components_1 - 1]),
            (z2.shape[1], elbo2.item(), kld2.item(), -recon_loss_2.item(), min_components_2, expl_var_2[min_components_2 - 1]),
            (z3.shape[1], elbo3.item(), kld3.item(), -recon_loss_3.item(), min_components_3, expl_var_3[min_components_3 - 1]),
        ]
    )

    df.columns = ["latent_size", "elbo", "kl divergence", "recon loss", "min components", "explained_variance"]

    df.to_csv('outs/vanilla_vae/3pca.csv')

def plot_pca_2d_map(root_path, id, category, include_site_assignments=False):
    import train
    from sklearn.decomposition import PCA
    from matplotlib import pyplot as plt

    vae = torch.load('{}/model_{}.pt'.format(root_path, id)) 
    test_dataloader = utils.get_adata_datasets(raw=True)[1].dataloader(128)
    _, _, _, z, cell_type, donor_id, site = train.test_a_dataset(vae, test_dataloader, limit_batches=1, include_site_assignments=include_site_assignments)

    pca = PCA(2)
    z_pca = pca.fit_transform(z)

    if category == 'cell_type':
        category_t = cell_type
    elif category == 'donor_id':
        category_t = donor_id
    elif category == 'site':
        category_t = site
    else:
        raise ValueError("unknown category: {}".format(category))

    plt.cla()
    plt.subplots(1, 1, figsize=(30,30))
    plt.scatter(z_pca[:, 0], z_pca[:, 1], c=category_t, linewidths=8)
    plt.savefig('{}/pca_scatter_{}_{}.png'.format(root_path, id, category))
    plt.cla()     

if __name__ == '__main__':
    print("1a")
    generate_summary_1()
    print("1b")
    generate_hists_1()
    print("1d")
    generate_hists_2()
    print("1f")
    generate_obs_summary()
    print("2a")        
    plot_vae("outs/vanilla_vae", "1")
    print("2b")
    plot_vae("outs/vanilla_vae", "2")
    plot_vae("outs/vanilla_vae", "3")
    generate_pca_table()
    print("2c")
    plot_pca_2d_map("outs/vanilla_vae", "1", category="cell_type")
    print("3b")
    plot_vae("outs/custom_vae", "1")
    print("3c")
    plot_pca_2d_map("outs/custom_vae", "1", category="cell_type")
    print("4a")
    plot_pca_2d_map("outs/vanilla_vae", "1", category="donor_id")
    plot_pca_2d_map("outs/vanilla_vae", "1", category="site")
    plot_pca_2d_map("outs/custom_vae", "1", category="donor_id")
    plot_pca_2d_map("outs/custom_vae", "1", category="site")
    print("4b")
    plot_vae("outs/custom_vae", "1s")
    plot_pca_2d_map("outs/custom_vae", "1s", category="cell_type", include_site_assignments=True)
