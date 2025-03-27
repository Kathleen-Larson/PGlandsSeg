import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

#------------------------------------------------------------------------------

class UNet3D(nn.Module):
    def __init__(self,
                 in_channels:int,             # no. of input channels
                 out_channels:int,            # no. out output features
                 conv_sz:int=3,               # conv window size
                 drop_rate:float=0.,          # drop out
                 pool_sz:int=2,               # pooling window size
                 n_convs_per_block:int=2,     # no. of layers per block
                 n_levels:int=4,              # no. of enc/dec levels
                 n_starting_features:int=24,  # no. of starting features
                 norm_func:str='Instance',    # normalization func.
                 activ_func:str='ELU',        # activation func.
                 pool_func:str='MaxPool',     # pooling func.
                 use_residuals:bool=False,    # flag for residual conns.
                 use_skips:bool=True,         # flag for skips conns.
                 X:int=3,                     # no. of spatial dims.
    ):
        """
        Main UNet class.
        """
        super(UNet3D, self).__init__()

        self.use_skips = use_skips
        self.n_levels = n_levels
        block_config = [n_starting_features * (2**i) for i in range(n_levels)]

        # Parse input functions
        activ_func = eval(f'{activ_func}')
        drop_func = torch.nn.Dropout(p=drop_rate) if drop_rate > 0 else None
        norm_func = eval(f'{norm_func}{X}d')

        if not callable(activ_func):
            utils.fatal(f'{activ_func} not a valid activation function')
        if not callable(norm_func):
            utils.fatal(f'{norm_func} not a valid normalization function')

        
        # Encoding blocks:
        self.encoding = nn.Sequential()
        encoding_config = [in_channels] + block_config
        
        for b in range(len(encoding_config)-1):
            block = _UNetBlock(n_input_features=encoding_config[b],
                               n_output_features=encoding_config[b+1],
                               n_layers=n_convs_per_block,
                               conv_sz=conv_sz,
                               norm=norm_func,
                               activ=activ_func,
                               level=b,
                               residuals=use_residuals,
                               drop=drop_func,
                               X=X,
            )
            self.encoding.add_module('EncodeBlock%d' % (b+1), block)
            if b != n_levels:
                pool = eval('nn.MaxPool%dd' % X)(kernel_size=pool_sz,
                                                 stride=pool_sz,
                                                 return_indices=True
                )
                self.encoding.add_module('Pool%d' % (b+1), pool)
                
            
        # Encoding blocks:
        self.decoding = nn.Sequential()
        decoding_config = [out_channels] + block_config
        
        for b in reversed(range(len(decoding_config)-1)):
            if b != n_levels:
                unpool = eval('nn.MaxUnpool%dd' % X)(kernel_size=pool_sz,
                                                     stride=pool_sz)
                self.decoding.add_module('Unpool%d' % (b+1), unpool)
            
            block = _UNetBlockTranspose(n_input_features=decoding_config[b+1],
                                        n_output_features=decoding_config[b],
                                        n_layers=n_convs_per_block,
                                        conv_sz=conv_sz,
                                        norm=norm_func,
                                        activ=activ_func,
                                        level=b,
                                        residuals=use_residuals,
                                        drop=drop_func,
                                        X=X,
            )
            self.decoding.add_module('DecodeBlock%d' % (b+1), block)

    ##
    def forward(self, x):
        enc = [None] * (self.n_levels + 1) # encoding
        dec = [None] * (self.n_levels) # decoding
        idx = [None] * (self.n_levels) # maxpool indices
        siz = [None] * (self.n_levels) # maxunpool output size
        
        # Encoding
        enc[0] = x
        for b in range(0, self.n_levels):
            x = enc[b+1] = \
                self.encoding.__getattr__('EncodeBlock%d' % (b+1))(x)
            siz[b] = x.shape            
            if b != self.n_levels - 1:
                x, idx[b] =  self.encoding.__getattr__('Pool%d' % (b+1))(x)

        # Decoding
        for b in reversed(range(self.n_levels)):
            if b != self.n_levels - 1:
                x = self.decoding.__getattr__('Unpool%d' % (b+1))\
                    (x, idx[b], output_size=siz[b])
            x = dec[b] = self.decoding.__getattr__('DecodeBlock%d' % (b+1))\
                (torch.cat([x, enc[b+1]], 1) if self.use_skips else x)

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
                 activ=None,
                 drop=None,
                 norm=None,
    ):
        """
        Encoding layer (called by _UNetBlock)
        """
        super(_UNetLayer, self).__init__()
        pad_sz = (conv_sz-1)//2 if conv_sz % 2 == 1 else (conv_sz//2)
        conv_bias = False if norm is not None else True

        activ = activ() if activ is not None else nn.Identity()
        conv = eval('nn.Conv%dd' % X)(n_input_features,
                                        n_output_features,
                                        kernel_size=conv_sz,
                                        padding=pad_sz,
                                        bias=conv_bias
        )
        drop = drop if drop is not None else nn.Identity()
        norm = norm(n_input_features) if norm is not None else nn.Identity()
        
        self.add_module('norm', norm)
        self.add_module('activ', activ)
        self.add_module('conv', conv)
        self.add_module('drop', drop)

    def forward(self, x):
        return self.drop(self.conv(self.activ(self.norm(x))))
      

###    
class _UNetLayerTranspose(nn.Module):
    def __init__(self,
                 X:int,
                 n_input_features:int,
                 n_output_features:int,
                 conv_sz:int=3,
                 activ=None,
                 norm=None,
                 drop=None,
    ):
        """
        Decoding layer (called by _UNetBlockTranspose)
        """
        super(_UNetLayerTranspose, self).__init__()
        pad_sz = (conv_sz-1)//2 if conv_sz % 2 == 1 else (conv_sz//2)
        conv_bias = False if norm is not None else True

        activ = activ() if activ is not None else nn.Identity()
        conv = eval('nn.ConvTranspose%dd' % X)(n_input_features,
                                               n_output_features,
                                               kernel_size=conv_sz,
                                               padding=pad_sz,
                                               bias=conv_bias
        )
        drop = drop if drop is not None else nn.Identity()
        norm = norm(n_input_features) if norm is not None else nn.Identity()

        self.add_module('drop', drop)
        self.add_module('norm', norm)
        self.add_module('activ', activ)
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
                 level:int,
                 norm=None,
                 activ=None,
                 residuals:bool=False,
                 drop=None,
                 skips=False,
                 **kwargs
    ):
        """
        Encoding UNet block
        """
        super(_UNetBlock, self).__init__()
        self.residuals = False if conv_sz % 2 == 0 else residuals

        for i in range(n_layers):
            n_in = n_input_features if i==0 else n_output_features
            n_out = (1 + (skips and i==(n_layers-1))) * n_output_features
            layer = _UNetLayer(n_input_features=n_in,
                               n_output_features=n_out,
                               conv_sz=conv_sz,
                               norm=norm,
                               activ=activ,
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
                 level:int,
                 norm=None,
                 activ=None,
                 residuals:bool=False,
                 drop=None,
                 skips=True,
                 **kwargs
    ):
        """
        Decoding UNet block
        """
        super(_UNetBlockTranspose, self).__init__()
        self.printout = False
        self.residuals = residuals if conv_sz % 2 == 0 else False

        for i in range(n_layers):
            n_in = (1 + (skips and i==0)) * n_input_features
            n_out = n_output_features if i==(n_layers-1) else n_input_features
            layer = _UNetLayerTranspose(n_input_features=n_in,
                                        n_output_features=n_out,
                                        conv_sz=conv_sz,
                                        norm=norm,
                                        activ=activ,
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
