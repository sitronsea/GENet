# Dataset
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from itertools import product

from . import utils


def get_label(W, n_dims, mode):
    prefactor = utils.prefactor(n_dims)
    flux_series_indices = utils.get_flux_series_indices(n_dims)
    if mode == "Berry":
        W = utils.unitary_log(W)

    label = torch.zeros(W.shape[0])
    for series in flux_series_indices:
        label += torch.einsum(
            "xii->x",
            utils.real_mul_chain(
                *[W[:, idx, :, :, 1] for idx in series]
            )
        )

    return prefactor * label


# Generate one random sample on the fly
def create_sample(n_bands, dims, label_mode="Berry", keep_only_trivial_samples=False):
    # Calculate several parameters
    num_sites = utils.site_prod(dims)
    n_dims = len(dims)
    num_ch = n_dims // 2 * (n_dims - 1)

    # Generate the links
    data_batch_W = torch.zeros(
        (num_sites, num_ch, n_bands, n_bands, 2), dtype=torch.float
    )
    while True:
        # Perform a random phase shift on the links
        data_batch_U = utils.random_U(num_sites, n_dims, n_bands=n_bands)

        # Generate the corresponding fluxes
        for mu_x in range(n_dims - 1):
            for mu_y in range(mu_x + 1, n_dims):
                channel_idx = utils.get_tuple_index(mu_x, mu_y, n_dims)
                data_batch_W[:, channel_idx] = utils.complex_mul_chain(
                    data_batch_U[:, mu_x],
                    utils.shift(
                        W=data_batch_U[:, mu_y],
                        axis=mu_x,
                        orientation=-1,
                        dims=dims
                    ),
                    utils.shift(
                        W=utils.h_conj_pseudo(data_batch_U[:, mu_x]),
                        axis=mu_y,
                        orientation=-1,
                        dims=dims
                    ),
                    utils.h_conj_pseudo(data_batch_U[:, mu_y])
                )
        # Get labels
        label = get_label(data_batch_W, n_dims, label_mode)

        if (not keep_only_trivial_samples) or torch.round(sum(label)) == 0:
            break

    data_batch = torch.cat((data_batch_U, data_batch_W), dim=1)
    return data_batch, label


def create_sample_diag(
    n_bands,
    dims,
    label_distribution,
    keep_only_trivial_samples
):
    # Calculate several parameters
    num_sites = utils.site_prod(dims)
    n_dims = len(dims)
    while True:
        label = utils.label_angle(num_sites, distribution=label_distribution)
        if (not keep_only_trivial_samples) or torch.round(sum(label)) == 0:
            label *= np.pi
            eigs_angle = utils.eig_angle(num_sites, n_bands, label)
            break

    data_batch_U = torch.zeros(
        (num_sites, n_dims, n_bands, n_bands, 2),
        dtype=torch.float
    )
    data_batch_W = torch.zeros(
        (num_sites, 1, n_bands, n_bands, 2),
        dtype=torch.float
    )
    data_batch_W[:, 0, :, :, 0] = torch.diag_embed(torch.cos(eigs_angle))
    data_batch_W[:, 0, :, :, 1] = torch.diag_embed(torch.sin(eigs_angle))

    data_batch = torch.cat((data_batch_U, data_batch_W), dim=1)
    return data_batch, label


def create_sample_hamiltonian(
    n_bands,
    dims
):
    Kx = dims[0]
    Ky = dims[1]
    N = 2 * n_bands
    Gamma = torch.diag(torch.tensor([1] * n_bands + [-1] * n_bands)) + 1j * torch.zeros(N, N)
    def random_Hcoeff(n_bands):
        A_real = torch.normal(0, 1, size=(n_bands, n_bands))
        A_imag = torch.normal(0, 1, size=(n_bands, n_bands))
        return 0.5 * (A_real + A_real.T + 1j * (A_imag - A_imag.T))
    def create_random_hamiltonian():
        pi = torch.pi
        P = 3
        
        kx = torch.linspace(-pi, pi, Kx)
        ky = torch.linspace(-pi, pi, Ky)
        kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing='ij') 

        H = 1j * torch.zeros(Kx, Ky, N, N)

        for px in range(-P, P + 1):
            for py in range(-P, P + 1):
                if max(abs(px), abs(py)) == 0:
                    continue 
                
                phase = px * kx_grid + py * ky_grid 

                A_p = random_Hcoeff(N)
                B_p = random_Hcoeff(N)

                sin_term = torch.sin(phase)[..., None, None] * A_p
                cos_term = torch.cos(phase)[..., None, None] * B_p

                H += sin_term + cos_term

        H = H - Gamma @ H @ Gamma

        return H
    
    Hm = create_random_hamiltonian()
    eigvals, eigvecs = torch.linalg.eigh(Hm)
    eigvecs = eigvecs.conj().transpose(-1, -2)
    occ_mask = eigvals < 0                                
    n_val = occ_mask.sum(-1).max().item()
    assert n_val == n_bands, "Wrong number of occupied bands"
    idx = eigvals.argsort(dim=-1)[..., :n_bands]           # 每个 k 取能量最低 n_bands
    V = torch.gather(eigvecs, -2, idx.unsqueeze(-2).expand(-1, -1, eigvecs.size(-1), -1))
    U_x = V.conj().transpose(-1, -2) @ torch.roll(V, shifts=-1, dims=0)
    U_y = V.conj().transpose(-1, -2) @ torch.roll(V, shifts=-1, dims=1)
    U_x, _ = torch.linalg.qr(U_x)
    U_y, _ = torch.linalg.qr(U_y)
    data_U = torch.stack((U_x, U_y), dim=2)
    data_W = U_x @ torch.roll(U_y, shifts=-1, dims=0) @ torch.roll(U_x, shifts=-1, dims=1).conj().transpose(-1, -2) @ U_y.conj().transpose(-1, -2)
    data_W = data_W.unsqueeze(2)
    def prefactor(n_dims):
        """
        Prefactor for calculating higher dimensional Chern numbers
        """
        n = n_dims // 2
        result = 1.
        for i in range(1, n + 1):
            result /= i
        result /= ((2 * torch.pi) ** n)
        return result
    data_batch = torch.cat((data_U, data_W), dim=2).view(-1, 3, n_bands, n_bands)
    data_batch = torch.stack((data_batch.real, data_batch.imag), dim=-1)
    label = get_label(data_batch[:, 2].unsqueeze(1), n_dims=2, mode="Berry")
    return data_batch, label


class ProjectDataset(Dataset):
    """
        Generates the data on the fly
    """

    def __init__(self, args):
        super().__init__()
        self.n_bands = args.n_bands
        self.dims = args.dims
        self.samples = args.samples
        self.diag_samples = round(args.diag_ratio * self.samples) * (len(self.dims) == 2)

        self.label_mode = args.label_mode
        self.distribution = args.label_distribution
        self.keep_only_trivial_samples = args.keep_only_trivial_samples

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        if idx < self.diag_samples:
            data, label = create_sample_diag(
                n_bands=self.n_bands,
                dims=self.dims,
                label_distribution=self.distribution,
                keep_only_trivial_samples=self.keep_only_trivial_samples
            )
        else:
            data, label = create_sample(
                n_bands=self.n_bands,
                dims=self.dims,
                label_mode=self.label_mode,
                keep_only_trivial_samples=self.keep_only_trivial_samples
            )

        return data, label.float()


class HamiltonianDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.n_bands = args.n_bands
        self.dims = args.dims
        self.samples = args.samples

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        data, label = create_sample_hamiltonian(
            n_bands=self.n_bands,
            dims=self.dims,
        )

        return data, label.float()