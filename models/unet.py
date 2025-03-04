import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 conv_window_size:int=3,
                 pool_window_size:int=2,
                 n_convs_per_block:int=2,
                 n_levels:int=3,
                 n_starting_features:int=24,
                 normalization_type:str='Instance',
                 activation_function:str='ELU',
                 pooling_type:str='MaxPool',
                 residuals:bool=False,
                 skip:bool=True,
                 X:int=3,
                 **kwargs
    ):
        super(UNet3D, self).__init__()

        self.conv_window_size = conv_window_size
        self.pool_window_size = pool_window_size

        self.block_config = [n_starting_features * (2**i) for i in range(n_levels)]
        self.n_blocks = len(self.block_config)
        self.skip = skip
        self.residuals = residuals
        
        self.activation_function = activation_function if callable(getattr(nn, activation_function)) \
            else Exception("Invalid activation_function (not an attribute of torch.nn")        

        # Encoding blocks:
        self.encoding = nn.Sequential()
        encoding_config = [in_channels] + self.block_config
        
        for b in range(len(encoding_config)-1):
            block = _UNetBlock(n_input_features=encoding_config[b],
                               n_output_features=encoding_config[b+1],
                               n_layers=n_convs_per_block,
                               conv_window_size=conv_window_size,
                               norm_type=normalization_type,
                               activ_fn=activation_function,
                               level=b,
                               residuals=residuals,
                               drop=0,
                               X=X,
            )
            self.encoding.add_module('EncodeBlock%d' % (b+1), block)
            if b != n_levels:
                pool = eval('nn.MaxPool%dd' % X)(kernel_size=pool_window_size,
                                                 stride=pool_window_size,
                                                 return_indices=True
                )
                self.encoding.add_module('Pool%d' % (b+1), pool)
                
            
        #Decoding blocks:
        self.decoding = nn.Sequential()
        decoding_config = [out_channels] + self.block_config
        
        for b in reversed(range(len(decoding_config)-1)):
            if b != n_levels:
                unpool = eval('nn.MaxUnpool%dd' % X)(kernel_size=pool_window_size,
                                                     stride=pool_window_size)
                self.decoding.add_module('Unpool%d' % (b+1), unpool)
            
            block = _UNetBlockTranspose(n_input_features=decoding_config[b+1],
                                        n_output_features=decoding_config[b],
                                        n_layers=n_convs_per_block,
                                        conv_window_size=conv_window_size,
                                        norm_type=normalization_type,
                                        activ_fn=activation_function,
                                        level=b,
                                        residuals=residuals,
                                        drop=0,
                                        X=X,
            )
            self.decoding.add_module('DecodeBlock%d' % (b+1), block)

        self.printout = False

        

    def forward(self, x):
        enc = [None] * (self.n_blocks + 1) # encoding
        dec = [None] * (self.n_blocks) # decoding
        idx = [None] * (self.n_blocks) # maxpool indices
        siz = [None] * (self.n_blocks) # maxunpool output size

        if self.printout: print("Input", ": ", [x.shape[i] for i in range(len(x.shape))])

        # Encoding
        enc[0] = x
        for b in range(0, self.n_blocks):
            x = enc[b+1] = self.encoding.__getattr__('EncodeBlock%d' % (b+1))(x)
            if self.printout: print('Downconv block %d' % (b+1), ": ", [x.shape[i] for i in range(len(x.shape))])
            siz[b] = x.shape

            if b != self.n_blocks - 1:
                x, idx[b] =  self.encoding.__getattr__('Pool%d' % (b+1))(x)
                if self.printout: print('Maxpool block %d' % (b+1), ": ", [x.shape[i] for i in range(len(x.shape))])

        # Decoding
        for b in reversed(range(self.n_blocks)):
            if b != self.n_blocks - 1:
                x = self.decoding.__getattr__('Unpool%d' % (b+1))(x, idx[b], output_size=siz[b])
                if self.printout: print('Maxunpool block %d' % (b+1), ": ", [x.shape[i] for i in range(len(x.shape))])

            if self.printout: print("Skip connection", ": ", [(torch.cat([x, enc[b+1]], 1).shape[i]) for i in range(len(x.shape))])
            x = dec[b] = self.decoding.__getattr__('DecodeBlock%d' % (b+1))(torch.cat([x, enc[b+1]], 1))
            if self.printout: print('Upconv block %d' % (b+1), ": ", [x.shape[i] for i in range(len(x.shape))])

        return x

            
        
class _UNetLayer(nn.Module):
    def __init__(self,
                 X:int,
                 n_input_features:int,
                 n_output_features:int,
                 conv_window_size:int=3,
                 norm_type:str=None,
                 activ_fn:str=None,
                 drop=0,
                 **kwargs
    ):
        super(_UNetLayer, self).__init__()
        self.printout = False
        pad_size = (conv_window_size-1)//2 if conv_window_size % 2 == 1 else (conv_window_size//2)
        
        normXd = eval('nn.%sNorm%dd' % (norm_type, X))(n_input_features)
        activXd = eval('nn.%s' % activ_fn)() if activ_fn is not None else nn.Identity()
        convXd = eval('nn.Conv%dd' % X)(n_input_features,
                                        n_output_features,
                                        kernel_size=conv_window_size,
                                        padding=pad_size,
                                        bias=False if norm_type is not None else True
        )
        dropXd = eval('nn.Dropout%dd' % X)
        
        self.add_module('norm', normXd if norm_type is not None else nn.Identity())
        self.add_module('activ', activXd if activ_fn is not None else nn.Identity())
        self.add_module('conv', convXd)
        self.add_module('drop', dropXd if drop > 0 else nn.Identity())

    def forward(self, x):        
        return self.drop(self.conv(self.activ(self.norm(x))))


    
class _UNetLayerTranspose(nn.Module):
    def __init__(self,
                 X:int,
                 n_input_features:int,
                 n_output_features:int,
                 conv_window_size:int=3,
                 norm_type:str=None,
                 activ_fn:str=None,
                 drop=0,
                 **kwargs
    ):
        super(_UNetLayerTranspose, self).__init__()
        self.printout = False
        pad_size = (conv_window_size-1)//2 if conv_window_size % 2 == 1 else (conv_window_size//2)
        
        dropXd = eval('nn.Dropout%dd' % X)
        normXd = eval('nn.%sNorm%dd' % (norm_type, X))(n_input_features)
        activXd = eval('nn.%s' % activ_fn)() if activ_fn is not None else nn.Identity()
        convXd = eval('nn.ConvTranspose%dd' % X)(n_input_features,
                                                 n_output_features,
                                                 kernel_size=conv_window_size,
                                                 padding=pad_size,
                                                 bias=False if norm_type is not None else True
        )

        self.add_module('drop', dropXd if drop > 0 else nn.Identity())
        self.add_module('norm', normXd if norm_type else nn.Identity())
        self.add_module('activ', activXd)
        self.add_module('conv', convXd)


    def forward(self, x):
        return self.conv(self.activ(self.norm(self.drop(x))))




class _UNetBlock(nn.ModuleDict):
    def __init__(self,
                 X:int,
                 n_input_features:int,
                 n_output_features:int,
                 n_layers:int,
                 conv_window_size:int,
                 norm_type:str,
                 activ_fn:str,
                 level:int,
                 residuals:bool=False,
                 drop=0,
                 skip=False,
                 **kwargs
    ):
        super(_UNetBlock, self).__init__()
        self.printout = False
        self.residuals = False if conv_window_size % 2 == 0 else residuals
        
        for i in range(n_layers):
            growth = 1 + (skip and i==(n_layers-1))
            layer = _UNetLayer(n_input_features=n_input_features if i==0 else n_output_features,
                               n_output_features=growth*n_output_features,
                               conv_window_size=conv_window_size,
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
            if self.printout: print(name, ": ", [x.shape[i] for i in range(len(x.shape))])
        return x


    
class _UNetBlockTranspose(nn.ModuleDict):
    def __init__(self,
                 X:int,
                 n_input_features:int,
                 n_output_features:int,
                 n_layers:int,
                 conv_window_size:int,
                 norm_type:str,
                 activ_fn:str,
                 level:int,
                 residuals:bool=False,
                 drop=0,
                 skip=True,
                 **kwargs
    ):
        super(_UNetBlockTranspose, self).__init__()
        self.printout = False
        self.residuals = residuals if conv_window_size % 2 == 0 else False
        
        for i in range(n_layers):
            growth = 1 + (skip and i==0)
            layer = _UNetLayerTranspose(n_input_features=growth*n_input_features,
                                        n_output_features=n_output_features if i==(n_layers-1) else n_input_features,
                                        conv_window_size=conv_window_size,
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
            if self.printout: print(name, ": ", [x.shape[i] for i in range(len(x.shape))])
        return x


    
class UNet3D_3levels(UNet3D):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         n_levels=3,
                         **kwargs
        )


class UNet3D_4levels(UNet3D):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         n_levels=4,
                         **kwargs
        )

        
class UNet3D_5levels(UNet3D):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         n_levels=5,
                         **kwargs
        )
