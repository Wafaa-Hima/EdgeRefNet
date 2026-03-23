import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
import functools
from einops import rearrange
import torchvision.models as models
import custom_models
from custom_models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d
from custom_models.edge_block import Edge_Encoder, Edge_Decoder
from custom_models.attention_block import HFAB
from custom_models.BIT import BIT
from custom_models.EGCTNet import EGCTNet
from custom_models.ChangeFormer import ChangeFormerV6
from custom_models.DTCDSCN import CDNet34 
from torchvision.models import resnet18, resnet34, resnet50


###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float((args.max_epochs * 2) + 1)
            lr_l = max(0.0, lr_l)
            return lr_l


        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs // 3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif args.lr_policy == 'cosine': 
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-7)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):

    elif args.net_G == 'BIT':
        net = BIT(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                  with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8)

    elif args.net_G == 'EGCTNet':
        net = EGCTNet(img_size=args.img_size, input_nc=3, output_nc=2, embed_dim=args.embed_dim, num_classes=args.n_class)
    
    elif args.net_G == 'ChangeFormer':
        net = ChangeFormerV6(embed_dim=256)  # ChangeFormer with Transformer Encoder and Convolutional Decoder (Fuse)
    
    elif args.net_G == "DTCDSCN":
        #The implementation of the paper"Building Change Detection for Remote Sensing Images Using a Dual Task Constrained Deep Siamese Convolutional Network Model "
        #Code copied from: https://github.com/fitzpchao/DTCDSCN
        net = CDNet34(in_channels=3)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)

###############################################################################
# main Functions
###############################################################################
class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False, False, False])
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False, True, True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False, True, True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=64, out_channels=output_nc)

        self.SA1 = HFAB(input_channel=64,input_size=128,ratio=0.5)
        self.SA2 = HFAB(input_channel=64,input_size=64,ratio=0.5)
        self.SA3 = HFAB(input_channel=128,input_size=32,ratio=0.5)
        self.SA4 = HFAB(input_channel=256,input_size=32,ratio=0.5)

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1, _ = self.forward_single(x1)
        x2, _ = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):

        # resnet layers
        edge_feature = []

        if x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        x = self.SA1(x)
        edge_feature.append(x)  # 1/2  64
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x)  # 1/4, in=64, out=64
        x_4 = self.SA2(x_4)
        edge_feature.append(x_4)  # 1/4  64

        x_8 = self.resnet.layer2(x_4)  # 1/8, in=64, out=128
        x_8 = self.SA3(x_8)
        edge_feature.append(x_8)  # 1/8  128
        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8)  # 1/8, in=128, out=256
            x_8 = self.SA4(x_8)
            edge_feature.append(x_8)  # 1/8  256
        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8)  # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)
        return x, edge_feature


class BASE_Transformer(ResNet):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """

    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, input_nc, output_nc, with_pos, resnet_stages_num=5,
                 token_len=4, token_trans=True, enc_depth=1, dec_depth=1, dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True, pool_mode='max', pool_size=2, backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None, with_decoder=True, num_heads=None):
        super(BASE_Transformer, self).__init__(input_nc, output_nc, backbone=backbone,
                                               resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )
        self.encoder = ResNet(input_nc=3, output_nc=1000)

        
        dim = 32 
        mlp_dim = 2 * dim  

        self.edge_attention = HFAB(
            input_channel=256, 
            input_size=256,
            ratio=0.25,
            reduction=32
        )
        
        self.transformer = Transformer(
            dim=dim,  
            depth=enc_depth,  
            heads=8,  
            dim_head=64,  
            mlp_dim=mlp_dim,  
        )
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.tokenizer = tokenizer
        if not self.tokenizer:
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder

        self.with_pos = with_pos
        if with_pos == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len * 2, 32))
        decoder_pos_size = 256 // 4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder = nn.Parameter(torch.randn(1, 32,
                                                                  decoder_pos_size,
                                                                  decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                                                      heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim,
                                                      dropout=0,
                                                      softmax=decoder_softmax)
        self.transformer_decoder_edge = TransformerDecoder(dim=dim, depth=self.dec_depth,
                                                           heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim,
                                                           dropout=0,
                                                           softmax=decoder_softmax)
        self.edge_encoder = Edge_Encoder()
        self.edge_decoder = Edge_Decoder(in_channels=32)
        self.edge_classifier = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, padding=1)
        self.HF = HFAB(input_channel=32, input_size=128, ratio=0.5)
        self.HF_ = HFAB(input_channel=32, input_size=128, ratio=0.5)
        self.HF__ = HFAB(input_channel=2, input_size=256, ratio=0.5, reduction=2)
        
        self.edge_adjust = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)  # تصغير الحجم إلى النصف
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.final_upsample = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(0.1) 
        )

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens

    def _forward_reshape_tokens(self, x):
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_transformer_decoder_edge(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder_edge(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def forward(self, x1, x2):
        # Forward backbone resnet
        x1, edge_feature1 = self.forward_single(x1)
        x2, edge_feature2 = self.forward_single(x2)

        # Edge processing
        edge1 = self.edge_encoder(edge_feature1[0], edge_feature1[1], edge_feature1[2], edge_feature1[3])
        edge2 = self.edge_encoder(edge_feature2[0], edge_feature2[1], edge_feature2[2], edge_feature2[3])
        
        edge1 = self.HF(edge1)
        edge2 = self.HF(edge2)
          
        edge_final1 = self.edge_decoder(edge1)
        edge_final2 = self.edge_decoder(edge2)
        edge_final1 = self.HF__(edge_final1)
        edge_final2 = self.HF__(edge_final2)
        edge_map = self.edge_classifier(torch.cat([edge_final1, edge_final2], dim=1))

        # Token processing
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
            
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
            
        # Decoder processing
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)
            x_edge1 = self._forward_transformer_decoder_edge(edge1, token1)
            x_edge2 = self._forward_transformer_decoder_edge(edge2, token2)

        # Adjust dimensions before concatenation
        x1 = self.upsamplex2(x1)  # [8,64,128,128]
        x2 = self.upsamplex2(x2)
        
        # Adjust edge features
        x_edge1 = self.edge_adjust(x_edge1)  # [8,64,128,128]
        x_edge2 = self.edge_adjust(x_edge2)

        # Ensure dimensions match
        if x1.size()[-2:] != x_edge1.size()[-2:]:
            x_edge1 = F.interpolate(x_edge1, size=x1.shape[-2:], mode='bilinear', align_corners=True)
            x_edge2 = F.interpolate(x_edge2, size=x2.shape[-2:], mode='bilinear', align_corners=True)

        # Final concatenation
        x1 = torch.cat([x1, x_edge1], dim=1)  # [8,64,128,128]
        x2 = torch.cat([x2, x_edge2], dim=1)
        print(f"x1 shape: {x1.shape}")
        print(f"x2 shape: {x2.shape}")
        
        # Feature differencing
        x = torch.abs(x1 - x2)
        x = self.upsamplex2(x)  # [8,64,256,256]
        x = self.downsample(x)
        x = self.final_upsample(x)
        
        print(f"x shape: {x.shape}")
        # Classification
        change_map = self.classifier(x)
        if self.output_sigmoid:
            change_map = self.sigmoid(change_map)
            
        output = [edge_map, change_map]
        return output

if __name__ == '__main__':
    a = torch.ones(8, 3, 256, 256)
    b = torch.ones(8, 3, 256, 256)
    c = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                         with_pos='learned')
    c(a, b)