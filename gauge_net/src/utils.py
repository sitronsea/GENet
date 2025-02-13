# Utilities
import torch


# Several computation tools for grids
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


def site_prod(dims):
    """
    Calculate the total number of grid points
    """
    prod = 1
    for dim in dims:
        prod *= dim

    return prod


def shift(W, axis, orientation, dims, batch_tensor=False):
    """
    For a tensor, whose grid axes are flattened, perform
    a shift along a certain axis
    """
    if batch_tensor:
        original_shape = tuple(W.shape)
        new_shape = (original_shape[0], *dims, *tuple(W.shape[2:]))

        W_shift = torch.roll(
            W.view(*new_shape),
            orientation, axis + 1
        ).view(*original_shape)

        return W_shift

    else:
        original_shape = tuple(W.shape)
        new_shape = (*dims, *tuple(W.shape[1:]))

        W_shift = torch.roll(
            W.view(*new_shape),
            orientation, axis
        ).view(*original_shape)

        return W_shift


def get_tuple_index(a, b, n=4):
    """
    Get the channel index for (a,b) in the flux tensor
    """
    assert a != b
    minid = min(a, b)
    maxid = max(a, b)

    return (2 * n - minid - 1) * (minid) // 2 + maxid - minid - 1


def get_flux_series_indices(n_dims):
    def get_ordered_perms(series):
        """
        Get permutations mu_1, nu_1, ..., mu_n, nu_n such that mu_i < nu_i
        """
        series = list(series)
        assert len(series) % 2 == 0
        if len(series) == 2:
            return [[min(series), max(series)]]

        perms = []
        for i in range(len(series) - 1):
            for j in range(i + 1, len(series)):
                current = [
                    min(series[i], series[j]),
                    max(series[i], series[j])
                ]
                remaining = series[:i] + series[i + 1:j] + series[j + 1:]
                for perm in get_ordered_perms(remaining):
                    perms.append(current + perm)

        return perms

    def get_perm_index(perm):
        assert len(perm) % 2 == 0
        indices = []
        for i in range(len(perm) // 2):
            indices.append(
                get_tuple_index(perm[2 * i], perm[2 * i + 1], n_dims)
            )
        return indices

    perms = get_ordered_perms(range(n_dims))
    series_indices = []
    for perm in perms:
        series_indices.append(get_perm_index(perm))

    return series_indices


# Several complex matrix operations
def h_conj(A):
    """
    Hermitian conjugation on cfloat matrices
    """
    return torch.conj(torch.transpose(A, -1, -2))


def h_conj_pseudo(A):
    """
    Hermitian conjugation on pseudo-cfloat matrices of the form A[...,n,n,2]
    """
    A = A.transpose(dim0=-2, dim1=-3).clone()
    A[..., 1] = -A[..., 1]
    return A


def complex_scalar_mul(A, B):
    Mul_Re = A[..., 0] * B[..., 0] - A[..., 1] * B[..., 1]
    Mul_Im = A[..., 0] * B[..., 1] + A[..., 1] * B[..., 0]
    return torch.stack((Mul_Re, Mul_Im), dim=-1)


def complex_mul(A, B):
    Mul_Re = A[..., 0] @ B[..., 0] - A[..., 1] @ B[..., 1]
    Mul_Im = A[..., 0] @ B[..., 1] + A[..., 1] @ B[..., 0]
    return torch.stack((Mul_Re, Mul_Im), dim=-1)


def complex_mul_chain(*matrices):
    result = matrices[0]
    for mat in matrices[1:]:
        result = complex_mul(result, mat)
    return result


def real_mul_chain(*matrices):
    result = matrices[0]
    for mat in matrices[1:]:
        result = result @ mat
    return result


def complex_einsum(pattern, a, b):
    aR, aI = a.select(-1, 0), a.select(-1, 1)
    bR, bI = b.select(-1, 0), b.select(-1, 1)

    cR = torch.einsum(pattern, aR, bR) - torch.einsum(pattern, aI, bI)
    cI = torch.einsum(pattern, aR, bI) + torch.einsum(pattern, aI, bR)
    c = torch.stack((cR, cI), dim=-1)
    return c


def unitary_log(W):
    W = W[..., 0] + 1j * W[..., 1]
    eig, V = torch.linalg.eig(W)
    log_eig = torch.diag(1j * eig.angles())
    logW = V @ log_eig @ h_conj(V)
    return torch.stack((logW.real, logW.imag), dim=-1)


# Several phase generation tools
def random_U(*shapes, n_bands):
    """
    Generate random U(n) matrices
    """
    A = torch.normal(0, 1, size=(*shapes, n_bands, n_bands)) + 1j * torch.normal(0, 1, size=(*shapes, n_bands, n_bands))
    q, r = torch.linalg.qr(A)

    # Force r to have real diagonal elements to give unique decomposition.
    diag_r = torch.diagonal(r, dim1=-2, dim2=-1)
    phase = diag_r / torch.abs(diag_r)
    phase[torch.isnan(phase)] = 1 + 0j
    phase_diag = torch.diag_embed(torch.conj(phase))
    q = q @ phase_diag

    return torch.stack((q.real, q.imag), dim=-1)


def random_angle(*shape):
    return torch.rand(*shape, 1) * 2 * torch.pi - torch.pi


def random_phase(*shape):
    phase = random_angle(*shape)
    return torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1)


def s1mod(x):
    """
    Modular function that restricts outputs to [-1,1)
    """
    x = x + 1
    x = x / 2
    x = x - torch.floor(x)
    x = x * 2
    x = x - 1
    return x


def label_angle(num_site, distribution=None):
    """
    Generate phase angles for fluxes
    """
    if distribution:
        n_partition = len(distribution)
        label = (torch.multinomial(torch.tensor(distribution), num_site - 1) + 0.5 + 0.5 * torch.randn(num_site - 1)) * 2 / (n_partition) - 1
        label = torch.cat(
            (label, -torch.sum(label, dim=0, keepdim=True)),
            dim=0
        )
        return s1mod(label)

    else:
        label = torch.rand(num_site - 1, 1) * 2 - 1
        label = torch.cat(
            (label, -torch.sum(label, dim=0, keepdim=True)),
            dim=0
        )
        return s1mod(label)


def eig_angle(num_sites, n_bands, labels):
    eigs = random_angle(num_sites, n_bands - 1)
    eigs = torch.cat((
        eigs,
        (labels - torch.sum(eigs, dim=1)).unsqueeze(-1)
    ), dim=1)
    return eigs


# Several tools for various layers
def var_init(
    init_variant,
    w_in_size,
    n_bands
):
    """
    Variance initialization for parameters
    """
    if init_variant == -1:
        variance = 1.0 / (2 * w_in_size)
    elif init_variant == 0:
        variance = 1.0 / (4 * w_in_size ** 2)
    elif init_variant == 1:
        variance = 1.0 / (n_bands ** 2 * w_in_size ** 2)
    elif init_variant == 2:
        variance = 1.0 / (8 * n_bands ** 2 * w_in_size ** 2)
    elif init_variant == 4:
        variance = 1.0 / (2.5 * w_in_size ** 2)
    else:
        print("init_variant {init_variant} not known, aborting")
        raise SystemExit(1)
    return variance


def unpack_x(x, n_dims):
    """
    Unpacking the links and the fluxes
    """
    u = x[:, :, 0:n_dims]
    w = x[:, :, n_dims:]
    return u, w


def repack_x(u, w):
    """
    Repacking the links and the fluxes (or different channels)
    """
    x = torch.cat((u, w), dim=2)
    return x


def transport(u, w, axis, orientation, dims):
    w_shift = shift(w, axis, orientation, dims, True)

    # select links of appropriate axis
    u_axis = u.select(dim=2, index=axis)

    if orientation < 0:
        w_terminal = complex_einsum(
            "bxij,bxwjk -> bxwik",
            u_axis,
            w_shift
        )
        w_terminal = complex_einsum(
            "bxwij,bxjk -> bxwik",
            w_terminal,
            h_conj_pseudo(u_axis)
        )

    else:
        # apply shift
        u_axis = shift(u_axis, axis, +1, dims, True)
        w_terminal = complex_einsum(
            "bxij,bxwjk -> bxwik",
            h_conj_pseudo(u_axis),
            w_shift
        )
        w_terminal = complex_einsum(
            "bxwij,bxjk -> bxwik",
            w_terminal,
            u_axis
        )

    return w_terminal


# Training and Evaluating
def rescale(outputs, labels):
    """
    Rescale outputs based on the labels
    """
    mean_fraction = torch.mean(labels / outputs)
    return outputs * mean_fraction, mean_fraction
