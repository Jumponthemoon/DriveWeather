import os
import gc
import copy
import lpips
import torch
import wandb
from glob import glob
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from diffusers.optimization import get_scheduler
from peft.utils import get_peft_model_state_dict
from cleanfid.fid import get_folder_features, build_feature_extractor, frechet_distance
import vision_aided_loss
from model import make_1step_sched
from cyclegan_turbo_3_weather_bc_processor import CycleGAN_Turbo, VAE_encode, VAE_decode, initialize_unet, initialize_vae
from my_utils.training_utils_3_weather import UnpairedDataset, build_transform, parse_args_unpaired_training
from my_utils.dino_struct import DinoStructureLoss
def initialize_discriminators():
    discriminators = {
        "snow": vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda"),
        "rain": vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda"),
        "fog": vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")

    }
    # Freeze feature extractors
    for discriminator in discriminators.values():
        discriminator.cv_ensemble.requires_grad_(False)
    return discriminators

def main(args):
    mode = 'train'
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with=args.report_to)
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer", revision=args.revision, use_fast=False,)
    noise_scheduler_1step = make_1step_sched()
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()

    unet, l_modules_unet_encoder, l_modules_unet_decoder, l_modules_unet_others = initialize_unet(args.lora_rank_unet, return_lora_module_names=True)
 
    vae_a2b, vae_lora_target_modules = initialize_vae(args.lora_rank_vae, return_lora_module_names=True)
    print(vae_a2b.encoder.conv_in.weight.shape[1],'-------------------------------')
    weight_dtype = torch.float32
    vae_a2b.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)

    if args.gan_disc_type == "vagan_clip":
        net_disc_snow = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        net_disc_snow.cv_ensemble.requires_grad_(False)  # Freeze feature extractor

        net_disc_rain = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        net_disc_rain.cv_ensemble.requires_grad_(False)  # Freeze feature extractor

        net_disc_fog = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        net_disc_fog.cv_ensemble.requires_grad_(False)  # Freeze feature extractor

        net_disc_b = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        net_disc_b.cv_ensemble.requires_grad_(False)  # Freeze feature extractor
 

    crit_cycle, crit_idt = torch.nn.L1Loss(), torch.nn.L1Loss()

    if args.enable_xformers_memory_efficient_attention:
        unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    unet.conv_in.requires_grad_(True)
    vae_b2a = copy.deepcopy(vae_a2b)
    params_gen = CycleGAN_Turbo.get_traininable_params(unet, vae_a2b, vae_b2a)

    vae_enc = VAE_encode(vae_a2b, vae_b2a=vae_b2a)
    vae_dec = VAE_decode(vae_a2b, vae_b2a=vae_b2a)

    optimizer_gen = torch.optim.AdamW(params_gen, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,)

  #  params_disc = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
    params_disc_snow = list(net_disc_snow.parameters()) + list(net_disc_b.parameters())
    params_disc_rain = list(net_disc_rain.parameters()) + list(net_disc_b.parameters())
    params_disc_fog = list(net_disc_fog.parameters()) + list(net_disc_b.parameters())


    # optimizer_disc = torch.optim.AdamW(params_disc, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
    #     weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,)

    optimizer_disc_snow = torch.optim.AdamW(params_disc_snow, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                            weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    optimizer_disc_rain = torch.optim.AdamW(params_disc_rain, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                            weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    optimizer_disc_fog = torch.optim.AdamW(params_disc_fog, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                            weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)


    dataset_train = UnpairedDataset(dataset_folder=args.dataset_folder, image_prep=args.train_img_prep, split="train", tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    T_val = build_transform(args.val_img_prep)
    fixed_caption_src = dataset_train.fixed_caption_src
    fixed_caption_tgt = dataset_train.fixed_caption_tgt
    l_images_src_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_src_test.extend(glob(os.path.join(args.dataset_folder, "test_A", ext)))
    l_images_tgt_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_tgt_test.extend(glob(os.path.join(args.dataset_folder, "test_B", ext)))
    l_images_src_test, l_images_tgt_test = sorted(l_images_src_test), sorted(l_images_tgt_test)


    lr_scheduler_gen = get_scheduler(args.lr_scheduler, optimizer=optimizer_gen,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)


    lr_scheduler_disc_snow = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc_snow,
                                        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                        num_training_steps=args.max_train_steps * accelerator.num_processes,
                                        num_cycles=args.lr_num_cycles, power=args.lr_power)
    lr_scheduler_disc_rain = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc_rain,
                                        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                        num_training_steps=args.max_train_steps * accelerator.num_processes,
                                        num_cycles=args.lr_num_cycles, power=args.lr_power)
    lr_scheduler_disc_fog = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc_fog,
                                        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                        num_training_steps=args.max_train_steps * accelerator.num_processes,
                                        num_cycles=args.lr_num_cycles, power=args.lr_power)


    net_lpips = lpips.LPIPS(net='vgg')
    net_lpips.cuda()
    net_lpips.requires_grad_(False)
    conditions = ['snow','rain','fog','night']

    fixed_caption_src = "Picture of a normal weather scene"
    fixed_caption_tgt = {condition:"Picture of a adverse weather scene" for condition in conditions}


    fixed_caption_tgt_night = "Picture of an adverse weather scene"

    # fixed_caption_tgt_snow = "Picture of an adverse weather scene"
    # fixed_caption_tgt_rain = "Picture of an adverse weather scene"
    # fixed_caption_tgt_fog = "Picture of an adverse weather scene"

    fixed_a2b_tokens = {}
    fixed_a2b_emb_base = {}
    for condition in conditions:
        tokens = tokenizer(fixed_caption_tgt[condition], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
        fixed_a2b_tokens[condition] = tokens
        fixed_a2b_emb_base[condition] = text_encoder(tokens.cuda().unsqueeze(0))[0].detach()


    fixed_a2b_tokens_night = tokenizer(fixed_caption_tgt_night, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
    fixed_a2b_emb_base_night = text_encoder(fixed_a2b_tokens_night.cuda().unsqueeze(0))[0].detach()

    fixed_a2b_tokens_snow = tokenizer(fixed_caption_tgt_snow, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
    fixed_a2b_emb_base_snow = text_encoder(fixed_a2b_tokens_snow.cuda().unsqueeze(0))[0].detach()

    fixed_a2b_tokens_rain = tokenizer(fixed_caption_tgt_rain, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
    fixed_a2b_emb_base_rain = text_encoder(fixed_a2b_tokens_rain.cuda().unsqueeze(0))[0].detach()

    fixed_a2b_tokens_fog = tokenizer(fixed_caption_tgt_fog, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
    fixed_a2b_emb_base_fog = text_encoder(fixed_a2b_tokens_fog.cuda().unsqueeze(0))[0].detach()

    fixed_b2a_tokens = tokenizer(fixed_caption_src, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
    fixed_b2a_emb_base = text_encoder(fixed_b2a_tokens.cuda().unsqueeze(0))[0].detach()
    del text_encoder, tokenizer  # free up some memory

    unet, vae_enc, vae_dec, net_disc_snow, net_disc_rain, net_disc_fog, net_disc_b = accelerator.prepare(unet, vae_enc, vae_dec, net_disc_snow, net_disc_rain, net_disc_fog, net_disc_b)




    net_lpips, optimizer_gen, optimizer_disc_snow, optimizer_disc_rain,optimizer_disc_fog, train_dataloader, lr_scheduler_gen, lr_scheduler_disc_snow, lr_scheduler_disc_rain, lr_scheduler_disc_fog= accelerator.prepare(
    net_lpips, optimizer_gen, optimizer_disc_snow, optimizer_disc_rain,optimizer_disc_fog, train_dataloader, lr_scheduler_gen, lr_scheduler_disc_snow, lr_scheduler_disc_rain, lr_scheduler_disc_fog
)




    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    first_epoch = 0
    global_step = 0
    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,)


    for name, module in net_disc_snow.named_modules():
        if "attn" in name:
            module.fused_attn = False
    for name, module in net_disc_rain.named_modules():
        if "attn" in name:
            module.fused_attn = False
    for name, module in net_disc_fog.named_modules():
        if "attn" in name:
            module.fused_attn = False

    for name, module in net_disc_b.named_modules():
        if "attn" in name:
            module.fused_attn = False

    for epoch in range(first_epoch, args.max_train_epochs):
        for step, batch in enumerate(train_dataloader):
           # l_acc = [unet, net_disc_a, net_disc_b, vae_enc, vae_dec]
            l_acc = [unet, net_disc_snow,net_disc_rain,net_disc_fog, net_disc_b, vae_enc, vae_dec]

            with accelerator.accumulate(*l_acc):
                img_a = batch["pixel_values_src"].to(dtype=weight_dtype)
                img_a_label = batch["pixel_values_src_label"].to(dtype=weight_dtype)
                img_b = batch["pixel_values_tgt"].to(dtype=weight_dtype)
                img_b_label = batch["pixel_values_tgt_label"].to(dtype=weight_dtype)

                tgt_weather = batch["tgt_weather"]

                bsz = img_a.shape[0]
                if tgt_weather[0] == 'snow':
                    # print(tgt_weather)
                    alpha = 1
                    fixed_a2b_emb = fixed_a2b_emb_base_snow.repeat(bsz, 1, 1).to(dtype=weight_dtype)
                    for p in unet.lora_params['foggy']:
                        p.requires_grad = False
                    for p in unet.lora_params['rainy']:
                        p.requires_grad = False
                    # 设置 `snowy` LoRA 参数为不可训练
                    for p in unet.lora_params['snowy']:
                        p.requires_grad = True
                elif tgt_weather[0] == 'rain':
                    # print(tgt_weather)
                    fixed_a2b_emb = fixed_a2b_emb_base_rain.repeat(bsz, 1, 1).to(dtype=weight_dtype)
                    alpha = 0
                    for p in unet.lora_params['foggy']:
                        p.requires_grad = False
                    for p in unet.lora_params['rainy']:
                        p.requires_grad = True
                    # 设置 `snowy` LoRA 参数为不可训练
                    for p in unet.lora_params['snowy']:
                        p.requires_grad = False
                elif tgt_weather[0] == 'fog':
                    fixed_a2b_emb = fixed_a2b_emb_base_fog.repeat(bsz, 1, 1).to(dtype=weight_dtype)
                    alpha = 0.5
                    for p in unet.lora_params['foggy']:
                        p.requires_grad = True
                    for p in unet.lora_params['rainy']:
                        p.requires_grad = False
                    # 设置 `snowy` LoRA 参数为不可训练
                    for p in unet.lora_params['snowy']:
                        p.requires_grad = False

                fixed_b2a_emb = fixed_b2a_emb_base.repeat(bsz, 1, 1).to(dtype=weight_dtype)
                timesteps = torch.tensor([noise_scheduler_1step.config.num_train_timesteps - 1] * bsz, device=img_a.device).long()

                """
                Cycle Objective
                """
              
                cyc_fake_b = CycleGAN_Turbo.forward_with_networks(img_a, img_a_label,"a2b", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_a2b_emb,alpha,mode)
                cyc_rec_a = CycleGAN_Turbo.forward_with_networks(cyc_fake_b, img_a_label,"b2a", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_b2a_emb,alpha,mode)

                loss_cycle_a = crit_cycle(cyc_rec_a, img_a) * args.lambda_cycle
                loss_cycle_a += net_lpips(cyc_rec_a, img_a).mean() * args.lambda_cycle_lpips
                # B -> fake A -> rec B
                cyc_fake_a = CycleGAN_Turbo.forward_with_networks(img_b, img_b_label,"b2a", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_b2a_emb,alpha,mode)
                cyc_rec_b = CycleGAN_Turbo.forward_with_networks(cyc_fake_a, img_b_label,"a2b", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_a2b_emb,alpha,mode)
                loss_cycle_b = crit_cycle(cyc_rec_b, img_b) * args.lambda_cycle
                loss_cycle_b += net_lpips(cyc_rec_b, img_b).mean() * args.lambda_cycle_lpips
                accelerator.backward(loss_cycle_a + loss_cycle_b, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
    
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                """
                Generator Objective (GAN) for task a->b and b->a (fake inputs)
                """
                fake_a = CycleGAN_Turbo.forward_with_networks(img_b, img_b_label, "b2a", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_b2a_emb,alpha,mode)
                fake_b = CycleGAN_Turbo.forward_with_networks(img_a, img_a_label, "a2b", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_a2b_emb,alpha,mode)
                #loss_gan_a = net_disc_a(fake_b, for_G=True).mean() * args.lambda_gan
                if tgt_weather[0] == 'snow':
                    loss_gan_a = net_disc_snow(fake_b, for_G=True).mean() * args.lambda_gan
                elif tgt_weather[0] == 'rain':
                    loss_gan_a = net_disc_rain(fake_b, for_G=True).mean() * args.lambda_gan
                elif tgt_weather[0] == 'fog':
                    loss_gan_a = net_disc_fog(fake_b, for_G=True).mean() * args.lambda_gan
                loss_gan_b = net_disc_b(fake_a, for_G=True).mean() * args.lambda_gan
                accelerator.backward(loss_gan_a + loss_gan_b, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                if tgt_weather[0] == 'snow':
                  optimizer_disc_snow.zero_grad()
                elif tgt_weather[0] == 'rain':
                  optimizer_disc_rain.zero_grad()
                elif tgt_weather[0] == 'fog':
                  optimizer_disc_fog.zero_grad()
                """
                Identity Objective
                """
                idt_a = CycleGAN_Turbo.forward_with_networks(img_b, img_b_label,"a2b", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_a2b_emb,alpha,mode)
                loss_idt_a = crit_idt(idt_a, img_b) * args.lambda_idt
                loss_idt_a += net_lpips(idt_a, img_b).mean() * args.lambda_idt_lpips
                idt_b = CycleGAN_Turbo.forward_with_networks(img_a, img_a_label,"b2a", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_b2a_emb,alpha,mode)
                loss_idt_b = crit_idt(idt_b, img_a) * args.lambda_idt
                loss_idt_b += net_lpips(idt_b, img_a).mean() * args.lambda_idt_lpips
                loss_g_idt = loss_idt_a + loss_idt_b
                accelerator.backward(loss_g_idt, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                """
                Discriminator for task a->b and b->a (fake inputs)
                """
                # loss_D_A_fake = net_disc_a(fake_b.detach(), for_real=False).mean() * args.lambda_gan
                if tgt_weather[0] =='snow':
                    loss_D_A_fake = net_disc_snow(fake_b.detach(), for_real=False).mean() * args.lambda_gan
                elif tgt_weather[0] =='rain':
                    loss_D_A_fake = net_disc_rain(fake_b.detach(), for_real=False).mean() * args.lambda_gan
                elif tgt_weather[0] =='fog':
                    loss_D_A_fake = net_disc_fog(fake_b.detach(), for_real=False).mean() * args.lambda_gan
                loss_D_B_fake = net_disc_b(fake_a.detach(), for_real=False).mean() * args.lambda_gan
                loss_D_fake = (loss_D_A_fake + loss_D_B_fake) * 0.5
                accelerator.backward(loss_D_fake, retain_graph=False)
                if accelerator.sync_gradients:
                    #params_to_clip = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
                    if tgt_weather[0] =='snow':
                        params_to_clip = list(net_disc_snow.parameters()) + list(net_disc_b.parameters()) 
                    elif tgt_weather[0] =='rain':
                        params_to_clip = list(net_disc_rain.parameters()) + list(net_disc_b.parameters()) 
                    elif tgt_weather[0] =='fog':
                        params_to_clip = list(net_disc_fog.parameters()) + list(net_disc_b.parameters()) 
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                if tgt_weather[0]=='snow':
                    optimizer_disc_snow.step()
                    lr_scheduler_disc_snow.step()
                    optimizer_disc_snow.zero_grad()
                elif tgt_weather[0]=='rain':
                    optimizer_disc_rain.step()
                    lr_scheduler_disc_rain.step()
                    optimizer_disc_rain.zero_grad()
                elif tgt_weather[0]=='fog':
                    optimizer_disc_fog.step()
                    lr_scheduler_disc_fog.step()
                    optimizer_disc_fog.zero_grad()
                """
                Discriminator for task a->b and b->a (real inputs)
                """
             #   loss_D_A_real = net_disc_a(img_b, for_real=True).mean() * args.lambda_gan
                if tgt_weather[0] =='snow':
                    loss_D_A_real = net_disc_snow(img_b, for_real=True).mean() * args.lambda_gan
                elif tgt_weather[0]=='rain':
                    loss_D_A_real = net_disc_rain(img_b, for_real=True).mean() * args.lambda_gan
                elif tgt_weather[0]=='fog':
                    loss_D_A_real = net_disc_fog(img_b, for_real=True).mean() * args.lambda_gan


                loss_D_B_real = net_disc_b(img_a, for_real=True).mean() * args.lambda_gan
                loss_D_real = (loss_D_A_real + loss_D_B_real) * 0.5
                accelerator.backward(loss_D_real, retain_graph=False)
                if accelerator.sync_gradients:
                    # params_to_clip = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
                    if tgt_weather[0]=='snow':
                        params_to_clip = list(net_disc_snow.parameters()) + list(net_disc_b.parameters())
                    elif tgt_weather[0]=='rain':
                        params_to_clip = list(net_disc_rain.parameters()) + list(net_disc_b.parameters())
                    elif tgt_weather[0]=='fog':
                        params_to_clip = list(net_disc_fog.parameters()) + list(net_disc_b.parameters())

                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                # optimizer_disc.step()
                # lr_scheduler_disc.step()
                # optimizer_disc.zero_grad()
                if tgt_weather[0]=='snow':
                    optimizer_disc_snow.step()
                    lr_scheduler_disc_snow.step()
                    optimizer_disc_snow.zero_grad()
                elif tgt_weather[0]=='rain':
                    optimizer_disc_rain.step()
                    lr_scheduler_disc_rain.step()
                    optimizer_disc_rain.zero_grad()
                elif tgt_weather[0]=='fog':
                    optimizer_disc_fog.step()
                    lr_scheduler_disc_fog.step()
                    optimizer_disc_fog.zero_grad()
            logs = {}
            logs["cycle_a"] = loss_cycle_a.detach().item()
            logs["cycle_b"] = loss_cycle_b.detach().item()
            logs["gan_a"] = loss_gan_a.detach().item()
            logs["gan_b"] = loss_gan_b.detach().item()
            logs["disc_a"] = loss_D_A_fake.detach().item() + loss_D_A_real.detach().item()
            logs["disc_b"] = loss_D_B_fake.detach().item() + loss_D_B_real.detach().item()
            logs["idt_a"] = loss_idt_a.detach().item()
            logs["idt_b"] = loss_idt_b.detach().item()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    eval_unet = accelerator.unwrap_model(unet)
                    eval_vae_enc = accelerator.unwrap_model(vae_enc)
                    eval_vae_dec = accelerator.unwrap_model(vae_dec)
                    if global_step % args.viz_freq == 1:
                        for tracker in accelerator.trackers:
                            if tracker.name == "wandb":
                                viz_img_a = batch["pixel_values_src"].to(dtype=weight_dtype)
                                viz_img_b = batch["pixel_values_tgt"].to(dtype=weight_dtype)
                                log_dict = {
                                    "train/real_a": [wandb.Image(viz_img_a[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
                                    "train/real_b": [wandb.Image(viz_img_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
                                }
                                log_dict["train/rec_a"] = [wandb.Image(cyc_rec_a[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)]
                                log_dict["train/rec_b"] = [wandb.Image(cyc_rec_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)]
                                log_dict["train/fake_b"] = [wandb.Image(fake_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)]
                                log_dict["train/fake_a"] = [wandb.Image(fake_a[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)]
                                tracker.log(log_dict)
                                gc.collect()
                                torch.cuda.empty_cache()

                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        sd = {}
                        sd["l_target_modules_encoder"] = l_modules_unet_encoder
                        sd["l_target_modules_decoder"] = l_modules_unet_decoder
                        sd["l_modules_others"] = l_modules_unet_others
                        sd["rank_unet"] = args.lora_rank_unet
              
                        sd["sd_snowy_encoder"] = get_peft_model_state_dict(eval_unet, adapter_name="snowy_encoder")
                        sd["sd_snowy_decoder"] = get_peft_model_state_dict(eval_unet, adapter_name="snowy_decoder")
                        sd["sd_snowy_other"] = get_peft_model_state_dict(eval_unet, adapter_name="snowy_others")
                        sd["sd_rainy_encoder"] = get_peft_model_state_dict(eval_unet, adapter_name="rainy_encoder")
                        sd["sd_rainy_decoder"] = get_peft_model_state_dict(eval_unet, adapter_name="rainy_decoder")
                        sd["sd_rainy_other"] = get_peft_model_state_dict(eval_unet, adapter_name="rainy_others")    

                        sd["sd_foggy_encoder"] = get_peft_model_state_dict(eval_unet, adapter_name="foggy_encoder")
                        sd["sd_foggy_decoder"] = get_peft_model_state_dict(eval_unet, adapter_name="foggy_decoder")
                        sd["sd_foggy_other"] = get_peft_model_state_dict(eval_unet, adapter_name="foggy_others")            
                        sd["rank_vae"] = args.lora_rank_vae
                        sd["vae_lora_target_modules"] = vae_lora_target_modules
                        sd["sd_vae_enc"] = eval_vae_enc.state_dict()
                        sd["sd_vae_dec"] = eval_vae_dec.state_dict()
                        torch.save(sd, outf)
                        gc.collect()
                        torch.cuda.empty_cache()

            

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break


if __name__ == "__main__":
    args = parse_args_unpaired_training()
    main(args)
