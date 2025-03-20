import torch
import torch.nn as nn
import torch.nn.functional as F

#------------------------------------------------------------------------------

class UNet3D(nn.Module):
    def __init__(self,
                 in_channels:int,                    # no. of input channels
                 out_channels:int,                   # no. out output features
                 conv_sz:int=3,                      # conv window size
                 pool_sz:int=2,                      # pooling window size
                 n_convs_per_block:int=2,            # no. of layers per block
                 n_levels:int=4,                     # no. of enc/dec levels
                 n_starting_features:int=24,         # no. of starting features
                 normalization_type:str='Instance',  # normalization func.
                 activation_function:str='ELU',      # activation func.
                 pooling_type:str='MaxPool',         # pooling func.
                 residuals:bool=False,               # flag for residual conns.
                 skip:bool=True,                     # flag for skip conns.
                 X:int=3,                            # no. of spatial dims.
    ):
        """
        Main UNet class.
        """
        super(UNet3D, self).__init__()

        self.conv_sz = conv_sz
        self.pool_sz = pool_sz

        self.block_config = [n_starting_features * (2**i) \
                             for i in range(n_levels)]
        self.n_blocks = len(self.block_config)
        self.skip = skip
        self.residuals = residuals
        
        self.activation_function = activation_function \
            if callable(getattr(nn, activation_function)) \
               else Exception('Invalid activation_function (not an ' +\
                              'attribute of torch.nn')

        # Encoding blocks:
        self.encoding = nn.Sequential()
        encoding_config = [in_channels] + self.block_config
        
        for b in range(len(encoding_config)-1):
            block = _UNetBlock(n_input_features=encoding_config[b],
                               n_output_features=encoding_config[b+1],
                               n_layers=n_convs_per_block,
                               conv_sz=conv_sz,
                               norm_type=normalization_type,
                               activ_fn=activation_function,
                               level=b,
                               residuals=residuals,
                               drop=0,
                               X=X,
            )
            self.encoding.add_module('EncodeBlock%d' % (b+1), block)
            if b != n_levels:
                pool = eval('nn.MaxPool%dd' % X)(kernel_size=pool_sz,
                                                 stride=pool_sz,
                                                 return_indices=True
                )
                self.encoding.add_module('Pool%d' % (b+1), pool)
                
            
        # Decoding blocks:
        self.decoding = nn.Sequential()
        decoding_config = [out_channels] + self.block_config
        
        for b in reversed(range(len(decoding_config)-1)):
            if b != n_levels:
                unpool = eval('nn.MaxUnpool%dd' % X)(kernel_size=pool_sz,
                                                     stride=pool_sz)
                self.decoding.add_module('Unpool%d' % (b+1), unpool)
            
            block = _UNetBlockTranspose(n_input_features=decoding_config[b+1],
                                        n_output_features=decoding_config[b],
                                        n_layers=n_convs_per_block,
                                        conv_sz=conv_sz,
                                        norm_type=normalization_type,
                                        activ_fn=activation_function,
                                        level=b,
                                        residuals=residuals,
                                        drop=0,
                                        X=X,
            )
            self.decoding.add_module('DecodeBlock%d' % (b+1), block)

    ##
    def forward(self, x):
        enc = [None] * (self.n_blocks + 1) # encoding
        dec = [None] * (self.n_blocks) # decoding
        idx = [None] * (self.n_blocks) # maxpool indices
        siz = [None] * (self.n_blocks) # maxunpool output size
        
        # Encoding
        enc[0] = x
        for b in range(0, self.n_blocks):
            x = enc[b+1] = \
                self.encoding.__getattr__('EncodeBlock%d' % (b+1))(x)
            siz[b] = x.shape            
            if b != self.n_blocks - 1:
                x, idx[b] =  self.encoding.__getattr__('Pool%d' % (b+1))(x)

        # Decoding
        for b in reversed(range(self.n_blocks)):
            if b != self.n_blocks - 1:
                x = self.decoding.__getattr__('Unpool%d' % (b+1))\
                    (x, idx[b], output_size=siz[b])
            x = dec[b] = self.decoding.__getattr__('DecodeBlock%d' % (b+1))\
                (torch.cat([x, enc[b+1]], 1))

        return x

            
#------------------------------------------------------------------------------
# UNet layer classes

###
class _UNetLayer(nn.Module):
    def __init__(self,
                 X:int,
                 n_input_features:int,
                 n_output_features:int,
                 conv_sz:int=3,
                 norm_type:str=None,
                 activ_fn:str=None,
                 drop_rate:float=0.,
                 **kwargs
    ):
        """
        Encoding layer (called by _UNetBlock)
        """
        super(_UNetLayer, self).__init__()
        pad_sz = (conv_sz-1)//2 if conv_sz % 2 == 1 else (conv_sz//2)
        conv_bias = False if norm_type is not None else True

        norm = eval('nn.%sNorm%dd' % (norm_type, X))(n_input_features)
        activ = eval('nn.%s' % activ_fn)() if activ_fn is not None \
            else nn.Identity()
        conv = eval('nn.Conv%dd' % X)(n_input_features,
                                        n_output_features,
                                        kernel_size=conv_sz,
                                        padding=pad_sz,
                                        bias=conv_bias
        )
        drop = eval('nn.Dropout%dd' % X)(p=drop_rate)

        self.add_module('norm', norm if norm_type is not None \
                        else nn.Identity())
        self.add_module('activ', activ if activ_fn is not None \
                        else nn.Identity())
        self.add_module('conv', conv)
        self.add_module('drop', drop if drop_rate > 0 else nn.Identity())

    def forward(self, x):
        return self.drop(self.conv(self.activ(self.norm(x))))
      

###    
class _UNetLayerTranspose(nn.Module):
    def __init__(self,
                 X:int,
                 n_input_features:int,
                 n_output_features:int,
                 conv_sz:int=3,
                 norm_type:str=None,
                 activ_fn:str=None,
                 drop_rate:float=0.,
                 **kwargs
    ):
        """
        Decoding layer (called by _UNetBlockTranspose)
        """
        super(_UNetLayerTranspose, self).__init__()
        pad_sz = (conv_sz-1)//2 if conv_sz % 2 == 1 else (conv_sz//2)
        conv_bias = False if norm_type is not None else True

        drop = eval('nn.Dropout%dd' % X)(p=drop_rate)
        norm = eval('nn.%sNorm%dd' % (norm_type, X))(n_input_features)
        activ = eval('nn.%s' % activ_fn)() if activ_fn is not None \
            else nn.Identity()
        conv = eval('nn.ConvTranspose%dd' % X)(n_input_features,
                                               n_output_features,
                                               kernel_size=conv_sz,
                                               padding=pad_sz,
                                               bias=conv_bias
        )

        self.add_module('drop', drop if drop_rate > 0 else nn.Identity())
        self.add_module('norm', norm if norm_type is not None \
                        else nn.Identity())
        self.add_module('activ', activ if activ_fn is not None \
                        else nn.Identity())
        self.add_module('conv', conv)

    def forward(self, x):
        return self.conv(self.activ(self.norm(self.drop(x))))



#------------------------------------------------------------------------------
# UNet block classes

###
class _UNetBlock(nn.ModuleDict):
    def __init__(self,
                 X:int,
                 n_input_features:int,
                 n_output_features:int,
                 n_layers:int,
                 conv_sz:int,
                 norm_type:str,
                 activ_fn:str,
                 level:int,
                 residuals:bool=False,
                 drop=0,
                 skip=False,
                 **kwargs
    ):
        """
        Encoding UNet block
        """
        super(_UNetBlock, self).__init__()
        self.residuals = False if conv_sz % 2 == 0 else residuals

        for i in range(n_layers):
            n_in = n_input_features if i==0 else n_output_features
            n_out = (1 + (skip and i==(n_layers-1))) * n_output_features
            layer = _UNetLayer(n_input_features=n_in,
                               n_output_features=n_out,
                               conv_sz=conv_sz,
                               norm_type=norm_type,
                               activ_fn=activ_fn,
                               drop=drop,
                               X=X,
            )
            self.add_module('ConvLayer%d' % (i+1), layer)

    def forward(self, x):
        for name, layer in self.items():
            res = x
            x = layer(x)
            if self.residuals and name[-1]!='1':  x += res
        return x


###
class _UNetBlockTranspose(nn.ModuleDict):
    def __init__(self,
                 X:int,
                 n_input_features:int,
                 n_output_features:int,
                 n_layers:int,
                 conv_sz:int,
                 norm_type:str,
                 activ_fn:str,
                 level:int,
                 residuals:bool=False,
                 drop=0,
                 skip=True,
                 **kwargs
    ):
        """
        Decoding UNet block
        """
        super(_UNetBlockTranspose, self).__init__()
        self.printout = False
        self.residuals = residuals if conv_sz % 2 == 0 else False

        for i in range(n_layers):
            n_in = (1 + (skip and i==0)) * n_input_features
            n_out = n_output_features if i==(n_layers-1) else n_input_features
            layer = _UNetLayerTranspose(n_input_features=n_in,
                                        n_output_features=n_out,
                                        conv_sz=conv_sz,
                                        norm_type=norm_type,
                                        activ_fn=activ_fn,
                                        drop=drop,
                                        X=X,
            )
            self.add_module('ConvLayer%d' % (i+1), layer)

    def forward(self, x):
        for name, layer in self.items():
            res = x
            x = layer(x)
            if self.residuals and name[-1]!='1':  x += res
        return x



#------------------------------------------------------------------------------
# Initializers w/ different numbers of levels

###
class UNet3D_3levels(UNet3D):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         n_levels=3,
                         **kwargs
        )

###
class UNet3D_4levels(UNet3D):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         n_levels=4,
                         **kwargs
        )

###     
class UNet3D_5levels(UNet3D):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         n_levels=5,
                         **kwargs
        )
