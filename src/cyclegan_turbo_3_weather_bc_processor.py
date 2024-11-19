import os
import sys
import copy
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd, download_url
from diffusers.models.attention_processor import AttnProcessor
from my_utils.cross_att import CrossViewAttnProcessor

class VAE_encode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_encode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        return _vae.encode(x).latent_dist.sample() * _vae.config.scaling_factor


class VAE_decode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_decode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        assert _vae.encoder.current_down_blocks is not None
        _vae.decoder.incoming_skip_acts = _vae.encoder.current_down_blocks
        x_decoded = (_vae.decode(x / _vae.config.scaling_factor).sample).clamp(-1, 1)
        return x_decoded

def initialize_unet(rank, return_lora_module_names=False):
    unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
    unet.requires_grad_(False)
    unet.train()
    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n: continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and "up_blocks" in n:
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break

    lora_conf_rainy_encoder = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_target_modules_encoder, lora_alpha=rank)
    lora_conf_rainy_decoder = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_target_modules_decoder, lora_alpha=rank)
    lora_conf_rainy_others = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_modules_others, lora_alpha=rank)
    
    lora_conf_snowy_encoder = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_target_modules_encoder, lora_alpha=rank)
    lora_conf_snowy_decoder = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_target_modules_decoder, lora_alpha=rank)
    lora_conf_snowy_others = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_modules_others, lora_alpha=rank)

    lora_conf_foggy_encoder = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_target_modules_encoder, lora_alpha=rank)
    lora_conf_foggy_decoder = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_target_modules_decoder, lora_alpha=rank)
    lora_conf_foggy_others = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_modules_others, lora_alpha=rank)

    lora_conf_night_encoder = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_target_modules_encoder, lora_alpha=rank)
    lora_conf_night_decoder = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_target_modules_decoder, lora_alpha=rank)
    lora_conf_night_others = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_modules_others, lora_alpha=rank)
    # 添加 `rainy` 风格的 LoRA 适配器
    unet.add_adapter(lora_conf_rainy_encoder, adapter_name="rainy_encoder")
    unet.add_adapter(lora_conf_rainy_decoder, adapter_name="rainy_decoder")
    unet.add_adapter(lora_conf_rainy_others, adapter_name="rainy_others")

    # 添加 `snowy` 风格的 LoRA 适配器
    unet.add_adapter(lora_conf_snowy_encoder, adapter_name="snowy_encoder")
    unet.add_adapter(lora_conf_snowy_decoder, adapter_name="snowy_decoder")
    unet.add_adapter(lora_conf_snowy_others, adapter_name="snowy_others")

    # 添加 `snowy` 风格的 LoRA 适配器
    unet.add_adapter(lora_conf_foggy_encoder, adapter_name="foggy_encoder")
    unet.add_adapter(lora_conf_foggy_decoder, adapter_name="foggy_decoder")
    unet.add_adapter(lora_conf_foggy_others, adapter_name="foggy_others")
   
    unet.add_adapter(lora_conf_night_encoder, adapter_name="night_encoder")
    unet.add_adapter(lora_conf_night_decoder, adapter_name="night_decoder")
    unet.add_adapter(lora_conf_night_others, adapter_name="night_others")
   
   # unet.set_adapters(["snowy_encoder", "snowy_decoder", "snowy_others","rainy_encoder", "rainy_decoder", "rainy_others"])

    if return_lora_module_names:
        return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
    else:
        return unet



def initialize_vae(rank=4, return_lora_module_names=False):
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
    old_conv_in = vae.encoder.conv_in

    # 设置新的输入通道数（从3增加到4）
    new_in_channels = old_conv_in.in_channels + 1  # 原为3，现在增加1变成4
    out_channels = old_conv_in.out_channels
    kernel_size = old_conv_in.kernel_size
    stride = old_conv_in.stride
    padding = old_conv_in.padding

    # 创建新的卷积层，增加输入通道
    new_conv_in = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )

    # 将原始卷积层的权重复制到新卷积层（前3个通道）
    with torch.no_grad():
        # 复制原始权重到前3个通道
        new_conv_in.weight[:, :old_conv_in.in_channels, :, :] = old_conv_in.weight
        # 初始化新增的通道为均值，或者全零也可以
        new_conv_in.weight[:, old_conv_in.in_channels:, :, :] = old_conv_in.weight.mean(dim=1, keepdim=True)
        # 复制bias参数
        new_conv_in.bias = old_conv_in.bias

    # 将VAE的conv_in替换为新的卷积层
    vae.encoder.conv_in = new_conv_in
    vae.requires_grad_(False)
    vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    vae.requires_grad_(True)
    vae.train()
    # add the skip connection convs
    vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1
    l_vae_target_modules = ["conv1","conv2","conv_in", "conv_shortcut",
        "conv", "conv_out", "skip_conv_1", "skip_conv_2", "skip_conv_3", 
        "skip_conv_4", "to_k", "to_q", "to_v", "to_out.0",
    ]
    vae_lora_config = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_vae_target_modules)
    vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
    if return_lora_module_names:
        return vae, l_vae_target_modules
    else:
        return vae


class CycleGAN_Turbo(torch.nn.Module):
    def __init__(self, mode = None, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_unet=8, lora_rank_vae=4):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
        # unet.set_attn_processor(
        #                processor=CrossViewAttnProcessor(self_attn_coeff=0.6,
        #                unet_chunk_size=1))
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False
        self.unet, self.vae = unet, vae
        if pretrained_name == "day_to_night":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/day2night.pkl"
            self.load_ckpt_from_url(url, ckpt_folder)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in the night"
            self.direction = "a2b"
        elif pretrained_name == "night_to_day":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/night2day.pkl"
            self.load_ckpt_from_url(url, ckpt_folder)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in the day"
            self.direction = "b2a"
        elif pretrained_name == "clear_to_rainy":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/clear2rainy.pkl"
            self.load_ckpt_from_url(url, ckpt_folder)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in heavy rain"
            self.direction = "a2b"
        elif pretrained_name == "rainy_to_clear":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/rainy2clear.pkl"
            self.load_ckpt_from_url(url, ckpt_folder)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in the day"
            self.direction = "b2a"
        
        elif pretrained_path is not None:
            sd = torch.load(pretrained_path)
            self.load_ckpt_from_state_dict(sd)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = None
            self.direction = None

        self.vae_enc.cuda()
        self.vae_dec.cuda()
        self.unet.cuda()

    def set_vae(self):
      #  self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
    # 设置新的输入通道数（从3增加到4）
        old_conv_in = self.vae.encoder.conv_in
        new_in_channels = old_conv_in.in_channels + 1  # 原为3，现在增加1变成4
        out_channels = old_conv_in.out_channels
        kernel_size = old_conv_in.kernel_size
        stride = old_conv_in.stride
        padding = old_conv_in.padding

        # 创建新的卷积层，增加输入通道
        new_conv_in = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # 将原始卷积层的权重复制到新卷积层（前3个通道）
        with torch.no_grad():
            # 复制原始权重到前3个通道
            new_conv_in.weight[:, :old_conv_in.in_channels, :, :] = old_conv_in.weight
            # 初始化新增的通道为均值，或者全零也可以
            new_conv_in.weight[:, old_conv_in.in_channels:, :, :] = old_conv_in.weight.mean(dim=1, keepdim=True)
            # 复制bias参数
            new_conv_in.bias = old_conv_in.bias

        # 将VAE的conv_in替换为新的卷积层
        self.vae.encoder.conv_in = new_conv_in
      #  return vae
    def set_unet(self):

        old_conv_in = self.unet.conv_in
        # 设置新的输入通道数（从3增加到4）
        new_in_channels = old_conv_in.in_channels + 1  # 原为3，现在增加1变成4
        out_channels = old_conv_in.out_channels
        kernel_size = old_conv_in.kernel_size
        stride = old_conv_in.stride
        padding = old_conv_in.padding

        # 创建新的卷积层，增加输入通道
        new_conv_in = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # 将原始卷积层的权重复制到新卷积层（前3个通道）
        with torch.no_grad():
            # 复制原始权重到前3个通道
            new_conv_in.weight[:, :old_conv_in.in_channels, :, :] = old_conv_in.weight
            # 初始化新增的通道为均值，或者全零也可以
            new_conv_in.weight[:, old_conv_in.in_channels:, :, :] = old_conv_in.weight.mean(dim=1, keepdim=True)
            # 复制bias参数
            new_conv_in.bias = old_conv_in.bias

        # 将UNet的conv_in替换为新的卷积层
        self.unet.conv_in = new_conv_in


    def extract_middle_image_attention(self, middle_image, caption_enc):
        # 提取中间图像的注意力特征
        self.middle_image_unet(middle_image, self.timesteps, caption_enc)
        return self.middle_image_unet.ref_att


    def load_ckpt_from_state_dict(self, sd):
        alpha = 0
        beta = 1
        theta = 0
        d = 1.1
        scale =1
        lora_conf_encoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_target_modules_encoder"], lora_alpha=sd["rank_unet"])
        lora_conf_decoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_target_modules_decoder"], lora_alpha=sd["rank_unet"])
        lora_conf_others = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_modules_others"], lora_alpha=sd["rank_unet"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_encoder.weight", ".weight")
            if "lora" in n and "default_encoder" in n:
                blended_weight = alpha * sd["sd_snowy_encoder"][name_sd] + beta * sd["sd_rainy_encoder"][name_sd]  + theta * sd["sd_foggy_encoder"][name_sd] 

                p.data.copy_(blended_weight)
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_decoder.weight", ".weight")
            if "lora" in n and "default_decoder" in n:
                blended_weight = alpha * sd["sd_snowy_decoder"][name_sd] + beta * sd["sd_rainy_decoder"][name_sd] + theta * sd["sd_foggy_decoder"][name_sd] 

                p.data.copy_(blended_weight)                
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_others.weight", ".weight")
            if "lora" in n and "default_others" in n:
                blended_weight = alpha * sd["sd_snowy_other"][name_sd]+ beta * sd["sd_rainy_other"][name_sd] + theta * sd["sd_foggy_other"][name_sd]

                p.data.copy_(blended_weight)

        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])


        self.set_vae()
        vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
        self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        self.vae.decoder.gamma = 1
        self.vae_b2a = copy.deepcopy(self.vae)
        self.vae_enc = VAE_encode(self.vae, vae_b2a=self.vae_b2a)
        self.vae_enc.load_state_dict(sd["sd_vae_enc"])
        self.vae_dec = VAE_decode(self.vae, vae_b2a=self.vae_b2a)

        self.vae_dec.load_state_dict(sd["sd_vae_dec"])

        self.vae_enc.vae.set_adapter(["vae_skip"])
        self.vae_enc.vae_b2a.set_adapter(["vae_skip"])

        self.vae_dec.vae.set_adapter(["vae_skip"])
        self.vae_dec.vae_b2a.set_adapter(["vae_skip"])

    def load_ckpt_from_url(self, url, ckpt_folder):
        os.makedirs(ckpt_folder, exist_ok=True)
        outf = os.path.join(ckpt_folder, os.path.basename(url))
        download_url(url, outf)
        sd = torch.load(outf)
        self.load_ckpt_from_state_dict(sd)

    @staticmethod
    # def forward_with_networks(x,x_label, direction, vae_enc, unet, vae_dec, sched, timesteps, text_emb,alpha,mode):
    def forward_with_networks(x, x_label, direction, vae_enc, unet, vae_dec, sched, timesteps, text_emb, alpha, mode, middle_image_attention_features=None):

        B = x.shape[0]
        assert direction in ["a2b", "b2a"]

        if mode == 'train':
            if alpha == 0:
                unet.set_adapters(["rainy_encoder", "rainy_decoder", "rainy_others"])

            elif alpha == 1:
                unet.set_adapters(["snowy_encoder", "snowy_decoder", "snowy_others"])
            elif alpha == 0.5:
                unet.set_adapters(["foggy_encoder", "foggy_decoder", "foggy_others"])

            else:
                unet.set_adapters(["night_encoder", "night_decoder", "night_others"])



        x_cat = torch.cat((x, x_label), dim=1)
        x_enc = vae_enc(x_cat, direction=direction).to(x.dtype)
        if mode !='train':
            text_emb =text_emb.repeat(10, 1, 1)
        #print(timesteps.size())
            timesteps = timesteps.repeat(10) 
      #  print(x_enc.size()) 
        model_pred = unet(x_enc, timesteps, encoder_hidden_states=text_emb,).sample
        x_out = torch.stack([sched.step(model_pred[i], timesteps[i], x_enc[i], return_dict=True).prev_sample for i in range(B)])
        x_out_decoded = vae_dec(x_out, direction=direction)
        x_out = x_out_decoded 
       # print(x_out.size(),'----')
        return x_out

    @staticmethod
    def get_traininable_params(unet, vae_a2b, vae_b2a):
        # add all unet parameters
        params_gen = list(unet.conv_in.parameters())
        unet.conv_in.requires_grad_(True)
        
        unet.set_adapters(["snowy_encoder", "snowy_decoder", "snowy_others","rainy_encoder", "rainy_decoder", "rainy_others","foggy_encoder", "foggy_decoder", "foggy_others","night_encoder", "night_decoder", "night_others"])
        unet.lora_params = {
            'rainy': [p for n, p in unet.named_parameters() if "rainy" in n and "lora" in n],
            'snowy': [p for n, p in unet.named_parameters() if "snowy" in n and "lora" in n],
            'foggy': [p for n, p in unet.named_parameters() if "foggy" in n and "lora" in n],
            'night': [p for n, p in unet.named_parameters() if "foggy" in n and "lora" in n]


        }
        params_gen += unet.lora_params['rainy'] + unet.lora_params['snowy'] +unet.lora_params['foggy'] +unet.lora_params['night']

 
        for n,p in vae_a2b.named_parameters():
            if "lora" in n and "vae_skip" in n:
                assert p.requires_grad
                params_gen.append(p)
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_1.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_2.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_3.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_4.parameters())

        # add all vae_b2a parameters
        for n,p in vae_b2a.named_parameters():
            if "lora" in n and "vae_skip" in n:
                assert p.requires_grad
                params_gen.append(p)
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_1.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_2.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_3.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_4.parameters())
        return params_gen

    def forward(self, x_t, x_t_label, alpha, mode=None, direction=None, caption=None, caption_emb=None, middle_image=None):

        if direction is None:
            assert self.direction is not None
            direction = self.direction
        if caption is None and caption_emb is None:
            assert self.caption is not None
            caption = self.caption
        if caption_emb is not None:
            caption_enc = caption_emb
        else:
            caption_tokens = self.tokenizer(caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt").input_ids.to(x_t.device)
            

            caption_enc = self.text_encoder(caption_tokens)[0].detach().clone()



        return self.forward_with_networks(
            x_t, x_t_label, direction, self.vae_enc, self.unet, self.vae_dec, self.sched, self.timesteps, caption_enc, alpha, mode)