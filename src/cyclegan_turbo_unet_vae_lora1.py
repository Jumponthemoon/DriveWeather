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
     #   _vae.set_adapter(["snowy_vae_skip"])
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
      #  _vae.set_adapter(["snowy_vae_skip"])
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
    # lora_conf_encoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder, lora_alpha=rank)
    # lora_conf_decoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder, lora_alpha=rank)
    # lora_conf_others = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_modules_others, lora_alpha=rank)
    # unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    # unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    # unet.add_adapter(lora_conf_others, adapter_name="default_others")
    # unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
    lora_conf_rainy_encoder = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_target_modules_encoder, lora_alpha=rank)
    lora_conf_rainy_decoder = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_target_modules_decoder, lora_alpha=rank)
    lora_conf_rainy_others = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_modules_others, lora_alpha=rank)
    
    lora_conf_snowy_encoder = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_target_modules_encoder, lora_alpha=rank)
    lora_conf_snowy_decoder = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_target_modules_decoder, lora_alpha=rank)
    lora_conf_snowy_others = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_modules_others, lora_alpha=rank)

    # 添加 `rainy` 风格的 LoRA 适配器
    unet.add_adapter(lora_conf_rainy_encoder, adapter_name="rainy_encoder")
    unet.add_adapter(lora_conf_rainy_decoder, adapter_name="rainy_decoder")
    unet.add_adapter(lora_conf_rainy_others, adapter_name="rainy_others")

    # 添加 `snowy` 风格的 LoRA 适配器
    unet.add_adapter(lora_conf_snowy_encoder, adapter_name="snowy_encoder")
    unet.add_adapter(lora_conf_snowy_decoder, adapter_name="snowy_decoder")
    unet.add_adapter(lora_conf_snowy_others, adapter_name="snowy_others")
   

   # unet.set_adapters(["snowy_encoder", "snowy_decoder", "snowy_others","rainy_encoder", "rainy_decoder", "rainy_others"])

    if return_lora_module_names:
        return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
    else:
        return unet


def initialize_vae(rank=4, return_lora_module_names=False):
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
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
    vae.add_adapter(vae_lora_config, adapter_name="snowy_vae_skip")
    vae.add_adapter(vae_lora_config, adapter_name="rainy_vae_skip")

    if return_lora_module_names:
        return vae, l_vae_target_modules
    else:
        return vae


class CycleGAN_Turbo(torch.nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_unet=8, lora_rank_vae=4):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
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

    def load_ckpt_from_state_dict(self, sd):
        alpha = 1
        lora_conf_encoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_target_modules_encoder"], lora_alpha=sd["rank_unet"])
        lora_conf_decoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_target_modules_decoder"], lora_alpha=sd["rank_unet"])
        lora_conf_others = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_modules_others"], lora_alpha=sd["rank_unet"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_encoder.weight", ".weight")
            if "lora" in n and "default_encoder" in n:
                blended_weight = alpha * sd["sd_snowy_encoder"][name_sd] + (1-alpha) * sd["sd_rainy_encoder"][name_sd]
                p.data.copy_(blended_weight)
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_decoder.weight", ".weight")
            if "lora" in n and "default_decoder" in n:
                blended_weight = alpha * sd["sd_snowy_decoder"][name_sd] + (1-alpha) * sd["sd_rainy_decoder"][name_sd]
                p.data.copy_(blended_weight)                
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_others.weight", ".weight")
            if "lora" in n and "default_others" in n:
                blended_weight = alpha * sd["sd_snowy_other"][name_sd] + (1-alpha) * sd["sd_rainy_other"][name_sd]
                p.data.copy_(blended_weight)

        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
        #self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")


        self.vae.add_adapter(vae_lora_config, adapter_name="snowy_vae_skip")
        self.vae.add_adapter(vae_lora_config, adapter_name="rainy_vae_skip")

       # self.vae.set_adapter(["snowy_vae_skip"])
        self.vae.decoder.gamma = 1
        self.vae_b2a = copy.deepcopy(self.vae)

        # print(sd["sd_vae_enc"].keys())
        self.vae_enc = VAE_encode(self.vae, vae_b2a=self.vae_b2a)
        # for n, p in self.vae_enc.named_parameters():
        #     if "lora" in n and "vae_skip" in n:
        #         # print('-----')

        #         # print(n)
        #         snowy_weight = n.replace('vae_skip','snowy_vae_skip')
        #         # print(snowy_weight)
        #         # print('-----')

        #         rainy_weight = n.replace('vae_skip','rainy_vae_skip')
        #         blended_weight = alpha * sd["sd_vae_enc"][snowy_weight] + (1-alpha) * sd["sd_vae_enc"][rainy_weight]
        #         p.data.copy_(blended_weight)
        #     else:
        #         p.data.copy_(sd["sd_vae_enc"][n])


        self.vae_enc.load_state_dict(sd["sd_vae_enc"])
        # for n, p in self.vae_enc.named_parameters():
        #     print(n)    
        # for n, p in sd["sd_vae_enc"].items():
        #     print(n)  
        self.vae_dec = VAE_decode(self.vae, vae_b2a=self.vae_b2a)
        # for n, p in self.vae_dec.named_parameters():
        #     if "lora" in n and "vae_skip" in n:
        #         snowy_weight = n.replace('vae_skip','snowy_vae_skip')
        #         rainy_weight = n.replace('vae_skip','rainy_vae_skip')
        #         blended_weight = alpha * sd["sd_vae_dec"][snowy_weight] + (1-alpha) * sd["sd_vae_dec"][rainy_weight]
        #         p.data.copy_(blended_weight)
        #     else:
        #         print(n)
        #         p.data.copy_(sd["sd_vae_dec"][n])
  #      self.vae_dec.load_state_dict(sd["sd_vae_dec"])
      
        self.vae_enc.vae.set_adapter(["snowy_vae_skip"])
        self.vae_enc.vae_b2a.set_adapter(["snowy_vae_skip"])

        self.vae_dec.vae.set_adapter(["snowy_vae_skip"])
        self.vae_dec.vae_b2a.set_adapter(["snowy_vae_skip"])


    def load_ckpt_from_url(self, url, ckpt_folder):
        os.makedirs(ckpt_folder, exist_ok=True)
        outf = os.path.join(ckpt_folder, os.path.basename(url))
        download_url(url, outf)
        sd = torch.load(outf)
        self.load_ckpt_from_state_dict(sd)

    @staticmethod
    def forward_with_networks(x, direction, vae_enc, unet, vae_dec, sched, timesteps, text_emb,alpha):
        B = x.shape[0]
        assert direction in ["a2b", "b2a"]

        # 根据 alpha 动态设置 UNet 的适配器


        # # Get LoRA state dicts for both 'rainy' and 'snowy'
        # rainy_state_dict = unet.get_adapter_state_dict("rainy")
        # snowy_state_dict = unet.get_adapter_state_dict("snowy")

        # # Blend the state dicts based on the alpha value
        # if 0 < alpha < 1:
        #     # Create a blended state dict
        #     blended_state_dict = {}
        #     for key in rainy_state_dict.keys():
        #         blended_state_dict[key] = (
        #             alpha * rainy_state_dict[key] + (1 - alpha) * snowy_state_dict[key]
        #         )
            
        #     # Load the blended state dict into the UNet's active weights
        #     unet.load_adapter_state_dict(blended_state_dict)
        # if alpha == 0:
        # #    print('set rainy')
        # #    unet.set_adapters(["rainy_encoder", "rainy_decoder", "rainy_others"])
        #     vae_enc.vae.set_adapter(["rainy_vae_skip"])
        #     vae_enc.vae_b2a.set_adapter(["rainy_vae_skip"])
        #     vae_dec.vae.set_adapter(["rainy_vae_skip"])
        #     vae_dec.vae_b2a.set_adapter(["rainy_vae_skip"])

        # elif alpha == 1:
        # #    print('set snowy')
        # #    unet.set_adapters(["snowy_encoder", "snowy_decoder", "snowy_others"])
        #     vae_enc.vae.set_adapter(["snowy_vae_skip"])
        #     vae_enc.vae_b2a.set_adapter(["snowy_vae_skip"])
        #     vae_dec.vae.set_adapter(["snowy_vae_skip"])
        #     vae_dec.vae_b2a.set_adapter(["snowy_vae_skip"])

        # else:
        #     raise print("Alpha should be between 0.0 and 1.0.")


        x_enc = vae_enc(x, direction=direction).to(x.dtype)
        model_pred = unet(x_enc, timesteps, encoder_hidden_states=text_emb,).sample
        x_out = torch.stack([sched.step(model_pred[i], timesteps[i], x_enc[i], return_dict=True).prev_sample for i in range(B)])
        x_out_decoded = vae_dec(x_out, direction=direction)
        return x_out_decoded

    @staticmethod
    def get_traininable_params(unet, vae_a2b, vae_b2a):
        # add all unet parameters
        params_gen = list(unet.conv_in.parameters())
        unet.conv_in.requires_grad_(True)
        
        unet.set_adapters(["snowy_encoder", "snowy_decoder", "snowy_others","rainy_encoder", "rainy_decoder", "rainy_others"])
        unet.lora_params = {
            'rainy': [p for n, p in unet.named_parameters() if "rainy" in n and "lora" in n],
            'snowy': [p for n, p in unet.named_parameters() if "snowy" in n and "lora" in n]
        }
        params_gen += unet.lora_params['rainy'] + unet.lora_params['snowy']

        # unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
        # for n,p in unet.named_parameters():
        #     if "lora" in n and "default" in n:
        #         assert p.requires_grad
        #         params_gen.append(p)
        # add all vae_a2b parameters
        vae_a2b.set_adapter(["snowy_vae_skip", "rainy_vae_skip"])
        # for n,p in vae_a2b.named_parameters():
        #     print(n)

        vae_a2b.lora_params = {
            'rainy': [p for n, p in vae_a2b.named_parameters() if "rainy" in n and "lora" in n],
            'snowy': [p for n, p in vae_a2b.named_parameters() if "snowy" in n and "lora" in n]
        }
        params_gen += vae_a2b.lora_params['rainy'] + vae_a2b.lora_params['snowy']
        # for n,p in vae_a2b.named_parameters():
        #     if "lora" in n and "vae_skip" in n:
        #         assert p.requires_grad
        #         params_gen.append(p)
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_1.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_2.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_3.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_4.parameters())

        # add all vae_b2a parameters
        vae_b2a.set_adapter(["snowy_vae_skip", "rainy_vae_skip"])
        vae_b2a.lora_params = {
            'rainy': [p for n, p in vae_b2a.named_parameters() if "rainy" in n and "lora" in n],
            'snowy': [p for n, p in vae_b2a.named_parameters() if "snowy" in n and "lora" in n]
        }
        params_gen += vae_b2a.lora_params['rainy'] + vae_b2a.lora_params['snowy']

        # for n,p in vae_b2a.named_parameters():
        #     if "lora" in n and "vae_skip" in n:
        #         assert p.requires_grad
        #         params_gen.append(p)
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_1.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_2.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_3.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_4.parameters())
        return params_gen

    def forward(self, x_t, alpha, direction=None, caption=None, caption_emb=None):
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

        # self.unet.set_adapters(["rainy_encoder", "rainy_decoder", "rainy_others"])
        # rainy_output = self.forward_with_networks(x_t, direction, self.vae_enc, self.unet, self.vae_dec, self.sched, self.timesteps, caption_enc)
        
        # self.unet.set_adapters(["snowy_encoder", "snowy_decoder", "snowy_others"])
        # snowy_output = self.forward_with_networks(x_t, direction, self.vae_enc, self.unet, self.vae_dec, self.sched, self.timesteps, caption_enc)
        
        # # 线性混合两个输出
        # combined_output = alpha * rainy_output + (1 - alpha) * snowy_output

        return self.forward_with_networks(
            x_t, direction, self.vae_enc, self.unet, self.vae_dec, self.sched, self.timesteps, caption_enc, alpha
        )