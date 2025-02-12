# Layers and network
import torch
import utils
# Below are the notations we use:

# Complex: w  =  bxvij
# b - batch
# x - lattice index
# v - layers
# i - matrix dim1
# j - matrix dim2

# Real: w  =  bxvijc
# b - batch
# x - lattice index
# v - layers
# i - matrix dim1
# j - matrix dim2
# c - real + complex parts

# Complex: u  =  bxvij
# b - batch
# x - lattice index
# v - orientation axis
# i - matrix dim1
# j - matrix dim2

# Complex: u  =  bxvijc
# b - batch
# x - lattice index
# v - orientation axis
# i - matrix dim1
# j - matrix dim2
# c - real + complex parts


# Parent class for layers
class GELayers(torch.nn.Module):
    def __init__(
        self,
        dims
    ):
        super().__init__()
        self.dims = dims
        self.update_dims(dims)

    def update_dims(self, dims):
        if len(dims) != len(self.dims):
            raise ValueError(
                "Length of new 'dims' must be the same as previous 'dims'."
            )
        self.dims = dims


# Layers for GEBLNet and GEConvNet
class GEConv(GELayers):
    """
    A module that performs gauge equivariant convolution
    """

    def __init__(
        self,
        dims,
        kernel_size,
        dilation,
        n_in,
        n_out,
        n_bands,
        use_unit_elements=True,
        conjugation_enlargement=True,
        init_w=1.0,
        init_variant=0,
    ):
        super().__init__(dims)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.n_in = n_in
        self.n_out = n_out
        self.n_bands = n_bands
        self.init_w = init_w
        self.use_unit_elements = use_unit_elements
        self.conjugation_enlargement = conjugation_enlargement

        # initialize weights
        n_in_size = self.n_in
        w_out_size = self.n_out

        if self.conjugation_enlargement:
            n_in_size *= 2
        if self.use_unit_elements:
            n_in_size += 1

        w_in_size = n_in_size * (self.kernel_size * len(self.dims) + 1)
        w_out_size = self.n_out

        self.weight = torch.nn.Parameter(
            data=torch.Tensor(w_out_size, w_in_size, 2),
            requires_grad=True
        )
        variance = utils.var_init(
            init_variant=init_variant,
            w_in_size=w_in_size,
            n_bands=self.n_bands
        )

        torch.nn.init.normal_(
            self.weight.data,
            std=init_w * torch.sqrt(variance)
        )

    def forward(self, x):
        # extract fluxes
        u, w = utils.unpack_x(x, len(self.dims))
        if self.conjugation_enlargement:
            w_c = utils.h_conj_pseudo(w)
            w = utils.repack_x(w, w_c)

        # enlarge tensors by unit matrices (adds bias and residual term)
        if self.use_unit_elements:
            unit_shape = list(w.shape)
            unit_shape[2] = 1
            unit_matrix = self.unit_matrix.to(w.device)
            unit_tensor = unit_matrix.expand(unit_shape)

            w = utils.repack_x(w, unit_tensor)

        # gather all terms along lattice axes up to kernel_size
        all_terms = []
        all_terms.append(w)
        for axis in range(len(self.dims)):
            w_transport = w.clone()
            for d in range(self.kernel_size):
                # get transported terms
                for step in range(self.dilation + 1):
                    w_transport = utils.transport(
                        u,
                        w_transport,
                        axis=axis,
                        orientation=-1,
                        dims=self.dims
                    )

                # and add to list
                all_terms.append(w_transport)

        # combine terms into a single tensor
        w = torch.cat(all_terms, dim=2)

        #  apply weights
        w = utils.complex_einsum("uv, bxvij -> bxuij", self.weight, w)

        return utils.repack_x(u, w)


class GEBL(GELayers):
    """
    A module that performs gauge equivariant bilinear layers
    """

    def __init__(
        self,
        dims,
        n_in,
        n_out,
        n_bands,
        use_unit_elements=True,
        conjugation_enlargement=True,
        init_w=1.0,
        init_variant=0,
    ):
        super().__init__(dims)
        self.n_in = n_in
        self.n_out = n_out
        self.n_bands = n_bands
        self.init_w = init_w
        self.use_unit_elements = use_unit_elements
        self.conjugation_enlargement = conjugation_enlargement

        # initialize weights
        w_in_size = self.n_in
        w_out_size = self.n_out

        if self.conjugation_enlargement:
            w_in_size *= 2
        if self.use_unit_elements:
            w_in_size += 1

        self.weight = torch.nn.Parameter(
            data=torch.Tensor(w_out_size, w_in_size, w_in_size, 2),
            requires_grad=True
        )
        variance = utils.var_init(
            init_variant=init_variant,
            w_in_size=w_in_size,
            n_bands=self.n_bands
        )
        torch.nn.init.normal_(
            self.weight.data,
            std=init_w * torch.sqrt(variance)
        )

        # construct unit matrix
        self.unit_matrix_re = torch.eye(self.n_bands)
        self.unit_matrix_im = torch.zeros_like(self.unit_matrix_re)
        self.unit_matrix = torch.stack(
            (self.unit_matrix_re, self.unit_matrix_im),
            dim=-1
        )

    def forward(self, x, y=None):
        if y is None:
            y = x.clone()

        u_x, w_x = utils.unpack_x(x, len(self.dims))
        _, w_y = utils.unpack_x(y, len(self.dims))

        # enlarge tensors by complex conjugates
        if self.conjugation_enlargement:
            w_cx = utils.h_conj_pseudo(w_x)
            w_cy = utils.h_conj_pseudo(w_y)
            w_x = utils.repack_x(w_x, w_cx)
            w_y = utils.repack_x(w_y, w_cy)

        # enlarge tensors by unit matrices (adds bias and residual term)
        if self.use_unit_elements:
            unit_shape = list(w_x.shape)
            unit_shape[2] = 1
            unit_matrix = self.unit_matrix.to(w_x.device)
            unit_tensor = unit_matrix.expand(unit_shape)

            w_x = utils.repack_x(w_x, unit_tensor)
            w_y = utils.repack_x(w_y, unit_tensor)

        # perform pair-wise multiplications and apply weights
        w_x = utils.complex_einsum("bxvij, bxwjk -> bxvwik", w_x, w_y)
        w_x = utils.complex_einsum("uvw, bxvwij -> bxuij", self.weight, w_x)

        return utils.repack_x(u_x, w_x)


class GEReLU(GELayers):
    """
    Computes ReLU( Re( Tr(W) ) ) W.
    This renders the output gauge-equivariant.
    """

    def __init__(self, dims):
        super().__init__(dims)

    def forward(self, x):

        # unpack
        u, w = utils.unpack_x(x, len(self.dims))

        # ReLU( Re( Tr(W) ) )
        relu_tr_w = torch.relu(
            torch.einsum("bxwiic -> bxwc", w).select(-1, 0)
        )

        w_res = torch.einsum("bxw, bxwijc -> bxwijc", relu_tr_w, w)

        return utils.repack_x(u, w_res)


class TrNorm(GELayers):
    """
    Normalize inputs based on traces across channels.
    This renders the output gauge-equivariant.
    """
    def __init__(self, dims, threshold, trnorm_on_abs):
        super().__init__(dims)
        self.threshold = threshold
        self.trnorm_on_abs = trnorm_on_abs

    def forward(self, x):
        u, w = utils.unpack_x(x, len(self.dims))
        tr = torch.einsum("bxviic->bxvc", w)
        if self.trnorm_on_abs:
            tr_abs = torch.sqrt(
                    tr[..., 0] ** 2 + tr[..., 1] ** 2
                )
            tr_abs_mean = torch.mean(tr_abs, dim=2)

            # clip values that are too small
            norm_factor = torch.clamp(tr_abs_mean, min=self.threshold)

        else:
            tr_mean = torch.mean(tr, dim=2)

            # clip values that are too small
            tr_mean_abs = torch.sqrt(
                tr_mean[..., 0] ** 2 + tr_mean[..., 1] ** 2
            )

            norm_factor = torch.clamp(tr_mean_abs, min=self.threshold)

        w_norm = torch.einsum(
                "bx, bxvijc->bxvijc",
                torch.reciprocal(norm_factor), w
                )

        return utils.repack_x(u, w_norm)


class Trace(GELayers):
    """
    Computes the trace of Wilson loops.
    This renders the gauge-invariant output.
    """

    def __init__(self, dims, normalize_trace):
        super().__init__(dims)
        self.normalize_trace = normalize_trace

    def forward(self, x):
        _, w = utils.unpack_x(x, len(self.dims))
        prefactor = 1.0 / w.shape[-2] if self.normalize_trace else 1.0
        tr = prefactor * torch.einsum("bxwiic -> bxwc", w)

        return tr


class Reshape(GELayers):
    """
        Reshapes input tensor.
    """

    def __init__(self, dims, shape):
        super().__init__(dims)
        self.shape = shape

    def forward(self, x):
        return x.view(tuple(self.shape))


class GEFluxOverlap(GELayers):
    """
    Overlapping outputs from GEBL and GEConv;
    a series of layers used in GEConvNet.
    """
    def __init__(
        self,
        dims,
        layer_channels,
        n_bands,
        dilation,
        kernel_size,
        use_unit_elements,
        conjugation_enlargement,
        init_w,
        init_variant
    ):
        super().__init__(dims)
        self.layer_channels = layer_channels
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.n_bands = n_bands
        self.use_unit_elements = use_unit_elements,
        self.conjugation_enlargement = conjugation_enlargement
        self.init_w = init_w
        self.init_variant = init_variant
        self.bl = GEBL(
            dims=self.dims,
            n_in=layer_channels[0],
            n_out=self.layer_channels[1],
            n_bands=self.n_bands,
            use_unit_elements=self.use_unit_elements,
            conjugation_enlargement=self.conjugation_enlargement,
            init_w=self.init_w,
            init_variant=self.init_variant
        )
        self.conv = GEConv(
            dims=self.dims,
            n_in=layer_channels[0],
            n_out=self.layer_channels[1],
            n_bands=self.n_bands,
            init_w=self.init_w,
            dilation=self.dilation,
            kernel_size=self.kernel_size
        )
        self.gerelu = GEReLU(dims=self.dims)
        self.bl_last = GEBL(
            dims=self.dims,
            n_in=layer_channels[1],
            n_out=self.layer_channels[2],
            n_bands=self.n_bands,
            init_w=self.init_w)

    def forward(self, x):
        x_bl = self.bl(x, x)
        x_bl = self.gerelu(x_bl)

        x_conv = self.conv(x)
        x_conv = self.gerelu(x_conv)

        return self.gerelu(self.bl_last(x_bl, x_conv))


# Layers for TrMLP
class TraceChar(torch.nn.Module):
    def __init__(self, channel):
        super(TraceChar, self).__init__()
        self.channel = channel

    def forward(self, x):
        _, w = utils.unpack_x(x, len(self.dims))
        w_original = w.clone()
        shape = w.shape
        device = w.device
        traces = [
            torch.stack(
                (
                    torch.full(shape[0:3], float(shape[3])),
                    torch.zeros(shape[0:3])
                ), dim=-1
            ).to(device)
        ]
        traces.append(torch.einsum("bxwiic->bxwc", w_original))
        for idx in range(self.channel - 1):
            w = utils.complex_mul(w_original, w)
            traces.append(torch.einsum("bxwiic->bxwc", w))

        x = torch.cat(traces, dim=2)
        return x


class ComplexLinear(torch.nn.Module):
    def __init__(self, input_channel, output_channel, init_w=2.):
        super(ComplexLinear, self).__init__()
        self.weight = torch.nn.Parameter(
            data=torch.Tensor(output_channel, input_channel, 2),
            requires_grad=True
        )
        torch.nn.init.normal_(
            self.weight.data,
            std=init_w / torch.sqrt(input_channel)
        )
        self.bias = torch.nn.Parameter(
            data=torch.zeros(output_channel, 2),
            requires_grad=True
        )

    def forward(self, x):
        return utils.complex_mul(self.weight, x) + self.bias


class ComplexAct(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x[..., 0]) * x


# Models
class GEBLNet(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        # Grid size and bands
        self.dims = args.dims
        self.num_sites = utils.site_prod(self.dims)
        self.n_dims = len(self.dims)
        self.n_bands = args.n_bands

        # Layer configurations
        # Layer channels
        self.input_channels = self.n_dims // 2 * (self.n_dims - 1)
        self.layer_channels = [self.input_channels] + args.layer_channels

        # GEBL layer configurations
        self.init_weight_factor = args.init_weight_factor
        self.init_variant = args.init_variant
        self.use_unit_elements = args.use_unit_elements
        self.conjugation_enlargement = args.conjugation_enlargement

        # TrNorm layer configurations
        self.trnorm = args.trnorm
        self.trnorm_threshold = args.trnorm_threshold
        self.trnorm_before_ReLU = args.trnorm_before_ReLU
        self.trnorm_on_abs = args.trnorm_on_abs

        # Trace layer configuration
        self.normalize_trace = args.normalize_trace

        # Layers
        # Activation
        self.gerelu = GEReLU(dims=self.dims)

        # Trace Normalization layer
        self.trnorm = TrNorm(
            dims=self.dims,
            threshold=self.trnorm_threshold,
            trnorm_on_abs=self.trnorm_on_abs
        )
        # Trace layer
        self.tr = Trace(
            dims=self.dims,
            normalize_trace=args.normalize_trace
        )
        # Final dense layer (linear)
        self.dense = torch.nn.Linear(self.layer_channels[-1] * 2, 1)

        # Assemble layers together
        self.layers = torch.nn.ModuleList()
        self.bls = torch.nn.ModuleList()

        for idx in range(len(self.layer_channels) - 1):
            # Bilinear layers
            bl = GEBL(
                dims=self.dims,
                n_in=self.layer_channels[idx],
                n_out=self.layer_channels[idx + 1],
                n_bands=self.n_bands,
                init_w=self.init_weight_factor,
                init_variant=args.init_variant,
                use_unit_elements=args.use_unit_elements
            )
            self.layers.append(bl)
            self.bls.append(bl)

            if self.trnorm:
                if self.trnorm_before_ReLU:
                    self.layers.append(self.trnorm)
                    self.layers.append(self.gerelu)

                else:
                    self.layers.append(self.gerelu)
                    self.layers.append(self.trnorm)

            else:
                self.layers.append(self.gerelu)

        self.layers.append(self.tr)

        self.layers.append(
            Reshape(
                dims=self.dims,
                shape=[-1, self.layer_channels[-1] * 2]
            )
        )
        self.layers.append(self.dense)
        self.layers.append(
            Reshape(
                dims=self.dims,
                shape=[-1, self.num_sites]
            )
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class GEConvNet(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        # Grid size and bands
        self.dims = args.dims
        self.num_sites = utils.site_prod(self.dims)
        self.n_dims = len(self.dims)
        self.n_bands = args.n_bands

        # Layer configurations
        # Layer channels
        self.input_channels = self.n_dims // 2 * (self.n_dims - 1)
        self.layer_channels = [self.input_channels] + args.layer_channels

        # GEBL and GEConv layer configurations
        self.init_weight_factor = args.init_weight_factor
        self.init_variant = args.init_variant
        self.use_unit_elements = args.use_unit_elements
        self.conjugation_enlargement = args.conjugation_enlargement
        self.dilation = args.dilation
        self.kernel_size = args.kernel_size

        # TrNorm layer configurations
        self.trnorm = args.trnorm
        self.trnorm_threshold = args.trnorm_threshold
        self.trnorm_before_ReLU = args.trnorm_before_ReLU
        self.trnorm_on_abs = args.trnorm_on_abs

        # Trace layer configuration
        self.normalize_trace = args.normalize_trace

        # Layers
        # Activation
        self.gerelu = GEReLU(dims=self.dims)

        # Trace Normalization layer
        self.trnorm = TrNorm(
            dims=self.dims,
            threshold=self.trnorm_threshold,
            trnorm_on_abs=self.trnorm_on_abs
        )
        # Trace layer
        self.tr = Trace(
            dims=self.dims,
            normalize_trace=args.normalize_trace
        )
        # Final dense layer (linear)
        self.dense = torch.nn.Linear(self.layer_channels[-1] * 2, 1)

        # Assemble layers together
        self.layers = torch.nn.ModuleList()

        for idx in range((len(self.layer_channels) - 1) // 2):
            self.fluxol = GEFluxOverlap(
                dims=self.dims,
                layer_channels=self.layer_channels[2 * idx:2 * idx + 3],
                n_bands=self.n_bands,
                dilation=self.dilation,
                kernel_size=self.kernel_size,
                use_unit_elements=self.use_unit_elements,
                conjugation_enlargement=self.conjugation_enlargement,
                init_w=self.init_weight_factor,
                init_variant=self.init_variant
            )
            if self.trnorm:
                self.layers.append(self.trnorm)

        self.layers.append(self.tr)

        self.layers.append(
            Reshape(
                dims=self.dims,
                shape=[-1, self.layer_channels[-1] * 2]
            )
        )
        self.layers.append(self.dense)
        self.layers.append(
            Reshape(
                dims=self.dims,
                shape=[-1, self.num_sites]
            )
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TrMLP(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        # Grid size and bands
        self.dims = args.dims
        self.num_sites = utils.site_prod(self.dims)
        self.n_dims = len(self.dims)
        self.n_bands = args.n_bands

        # Layer configurations
        # Layer channels
        self.input_channels = self.n_dims // 2 * (self.n_dims - 1)
        self.layer_channels = [self.input_channels] + args.layer_channels

        # Layers
        # Extract TrW^n
        self.tracechar = TraceChar(
            channel=self.layer_channels[0]
        )
        # Last dense layer
        self.dense = torch.nn.Linear(self.layer_channels[-1] * 2, 1)

        # Assemble layers together
        self.layers = torch.nn.ModuleList()
        self.layers.append(self.tracechar)

        for idx in range(len(self.layer_channels) - 1):
            self.layers.append(
                ComplexLinear(
                    input_channel=self.layer_channels[idx],
                    output_channel=self.layer_channels[idx + 1]
                )
            )
            self.layers.append(ComplexAct())

        self.layers.append(
            Reshape(
                dims=self.dims,
                shape=[-1, self.layer_channels[-1] * 2]
            )
        )
        self.layers.append(self.layer_channels[-1] * 2)
        self.layers.append(
            Reshape(
                dims=self.dims,
                shape=[-1, self.num_sites]
            )
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
