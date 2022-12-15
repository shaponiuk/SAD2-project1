import model
import utils
import torch


def train_epoch(model, dataloader, optimizer, device='cpu', include_site_assignments=False):
    device = torch.device(device)
    model = model.to(device)
    train_elbo = 0
    train_Dkl = 0
    train_recon_loss = 0
    num_batches = len(dataloader)

    for batch, (X, _, _, site,) in enumerate(dataloader):
        X = X.to(device)
        site_assignments = None
        if include_site_assignments:
            site_assignments = torch.nn.functional.one_hot(site, 4).float().to(device)
        elbo, Dkl, recon_loss, _, X_pred = model(X, site_assignments)
        mse = ((X - X_pred)**2).mean()
        optimizer.zero_grad()
        elbo.backward()
        optimizer.step()
        train_elbo += elbo
        train_Dkl += Dkl
        train_recon_loss += recon_loss
        
        print(
            "elbo: {:4f} recon_loss: {:4f} dkl: {:4f} mse: {:4f} {}/{}        ".format(
                elbo.item(), -recon_loss.item(), Dkl.item(), mse.item(), batch + 1, num_batches
            ),
            end='\r',
        )
            
    print()
            
    return (train_elbo/num_batches, 
            train_Dkl/num_batches, 
            train_recon_loss/num_batches)


def test_a_dataset(model, dataloader, device='cpu', limit_batches=None, include_site_assignments=False):
    device = torch.device(device)
    model = model.to(device)

    elbos = []
    klds = []
    recon_losses = []

    out_zs = []
    out_cell_types = []
    out_donor_ids = []
    out_sites = []

    with torch.no_grad():
        for batch, (X, cell_type, donor_id, site,) in enumerate(dataloader):
            if limit_batches == batch:
                break
            X = X.to(device)
            site_assignments = None
            if include_site_assignments:
                site_assignments = torch.nn.functional.one_hot(site, 4).float().to(device)
            elbo, kld, recon_loss, z, _ = model(X, site_assignments)
            out_zs.append(z.cpu())
            out_cell_types.append(cell_type)
            out_donor_ids.append(donor_id)
            out_sites.append(site)
            elbos.append(elbo.view(1))
            klds.append(kld.view(1))
            recon_losses.append(recon_loss.view(1))

    elbos = torch.cat(elbos, dim=0).mean()
    klds = torch.cat(klds, dim=0).mean()
    recon_losses = torch.cat(recon_losses, dim=0).mean()
    out_zs = torch.cat(out_zs, dim=0)
    out_cell_types = torch.cat(out_cell_types, dim=0)
    out_donor_ids = torch.cat(out_donor_ids, dim=0)
    out_sites = torch.cat(out_sites, dim=0)

    return elbos, klds, recon_losses, out_zs, out_cell_types, out_donor_ids, out_cell_types

def vae_training_loop(
    train_dataloader, 
    test_dataloader,
    vae, 
    optim, 
    epochs, 
    device, 
    include_site_assignments,
):

    train_elbo = []
    test_elbo = []
    train_kld = []
    test_kld = []
    train_recon_loss = []
    test_recon_loss = []

    for t in range(epochs):
        print("epoch:", t+1)
        elbo_train_, Dkl_train_, recon_loss_train_, = train_epoch(vae, train_dataloader, optim, device, include_site_assignments)
        train_elbo.append(elbo_train_)
        train_kld.append(Dkl_train_)
        train_recon_loss.append(recon_loss_train_)
        elbo_test_, kld_test_, recon_loss_test_, z, cell_types, _, _ = test_a_dataset(
            vae, 
            test_dataloader, 
            device, 
            include_site_assignments=include_site_assignments
        )
        test_elbo.append(elbo_test_)
        test_kld.append(kld_test_)
        test_recon_loss.append(recon_loss_test_)

    return {
        'train_elbo': train_elbo,
        'test_elbo': test_elbo,
        'train_kld': train_kld,
        'test_kld': test_kld,
        'train_recon_loss': train_recon_loss,
        'test_recon_loss': test_recon_loss,
    }, z, cell_types

def train_vanilla_vae_1(
    epochs,
    batch_size,
    latent_size,
    hidden_size,
    lr,
    id,
):
    train_dataset, test_dataset = utils.get_adata_datasets(raw=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader = train_dataset.dataloader(batch_size)
    test_dataloader = test_dataset.dataloader(batch_size)

    obs_dim = train_dataset.data.shape[1]

    enc = model.EncGaussian(model.EncNet(obs_dim, latent_size, hidden_size))
    dec = model.DecGaussian(model.DecNet(latent_size, obs_dim, hidden_size))
    vae = model.VAE(enc, dec)
            
    optim = torch.optim.Adam(params=vae.parameters(), lr=lr)
        
    results, _, _ = vae_training_loop(
        train_dataloader, test_dataloader,
        vae, optim, epochs, device, include_site_assignments=False,
    )

    utils.create_out_dir('./outs/vanilla_vae')
    torch.save(vae, 'outs/vanilla_vae/model_{}.pt'.format(id))
    torch.save(results, 'outs/vanilla_vae/model_{}_epochs.pt'.format(id))

def train_custom_vae_1(
    epochs,
    batch_size,
    latent_size,
    hidden_size,
    lr,
    id,
    include_site_assignments=False
):
    train_dataset, test_dataset = utils.get_adata_datasets(raw=True, normalise=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader = train_dataset.dataloader(batch_size)
    test_dataloader = test_dataset.dataloader(batch_size)

    obs_dim = train_dataset.data.shape[1]

    enc = model.EncGaussian(model.EncNet(obs_dim, latent_size, hidden_size))
    dec = model.DecNegBinom(model.DecNet(latent_size, obs_dim, hidden_size))
    vae = model.VAE(enc, dec)
            
    optim = torch.optim.Adam(params=vae.parameters(), lr=lr)
        
    results, _, _ = vae_training_loop(
        train_dataloader, test_dataloader,
        vae, optim, epochs, device, include_site_assignments=include_site_assignments
    )

    utils.create_out_dir('./outs/custom_vae')
    torch.save(vae, 'outs/custom_vae/model_{}.pt'.format(id))
    torch.save(results, 'outs/custom_vae/model_{}_epochs.pt'.format(id))

def train_vanilla():
    train_vanilla_vae_1(
        epochs = 10,
        batch_size = 128,
        latent_size = 32,
        hidden_size = 64,
        lr = 5 * 1e-3,
        id = '1',
    )

    train_vanilla_vae_1(
        epochs = 10,
        batch_size = 128,
        latent_size = 16,
        hidden_size = 64,
        lr = 5 * 1e-3,
        id = '2',
    )

    train_vanilla_vae_1(
        epochs = 10,
        batch_size = 128,
        latent_size = 8,
        hidden_size = 64,
        lr = 5 * 1e-3,
        id = '3',
    )

def train_custom():
    train_custom_vae_1(
        epochs = 10,
        batch_size = 128,
        latent_size = 32,
        hidden_size = 64,
        lr = 5 * 1e-3,
        id = '1'
    )

def train_site():
    train_custom_vae_1(
        epochs = 10,
        batch_size = 128,
        latent_size = 32,
        hidden_size = 64,
        lr = 5 * 1e-3,
        id = '1s',
        include_site_assignments = False,
    )

if __name__ == '__main__':
    print("training vanilla")
    train_vanilla()
    print("training custom")
    train_custom()
    print("training")
    train_site()