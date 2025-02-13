# Dataset
import torch
from torch.utils.data.dataset import Dataset

import utils


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
def create_sample(n_bands, dims, label_mode, keep_only_trivial_samples):
    # Calculate several parameters
    num_sites = utils.site_prod(dims)
    n_dims = len(dims)
    num_ch = n_dims // 2 * (n_dims - 1)

    # Generate the links
    data_batch_U = utils.random_U(num_sites, n_dims, n_bands=n_bands)
    data_batch_W = torch.zeros(
        (num_sites, num_ch, n_bands, n_bands, 2), dtype=torch.float
    )
    while True:
        # Perform a random phase shift on the links
        rand_phase = utils.random_phase(num_sites, n_dims)
        data_batch_U = utils.complex_scalar_mul(rand_phase, data_batch_U)

        # Generate the corresponding fluxes
        for mu_x in range(n_dims - 1):
            for mu_y in range(mu_x + 1, n_dims):
                channel_idx = utils.get_tuple_index(mu_x, mu_y, n_dims)
                data_batch_W[:, channel_idx] = utils.complex_mul_chain(
                    data_batch_U[:, mu_x],
                    utils.shift(
                        a=data_batch_U[:, mu_y],
                        axis=mu_x,
                        orientation=-1,
                        dims=dims
                    ),
                    utils.shift(
                        a=utils.h_conj_pseudo(data_batch_U[:, mu_x]),
                        axis=mu_y,
                        orientation=-1,
                        dims=dims
                    ),
                    utils.h_conj_pseudo(data_batch_U[:, mu_y])
                )
        # Get labels
        label = get_label(data_batch_W, n_dims, label_mode)

        if (not keep_only_trivial_samples) or round(sum(label)) == 0:
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
        if (not keep_only_trivial_samples) or round(sum(label)) == 0:
            label *= torch.pi
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
