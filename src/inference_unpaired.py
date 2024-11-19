import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from cyclegan_turbo_3_weather_no_skip import CycleGAN_Turbo
from my_utils.training_utils_3_weather import build_transform


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help='path to the input image')
    parser.add_argument('--prompt', type=str, required=False, help='the prompt to be used. It is required when loading a custom model_path.')
    parser.add_argument('--model_name', type=str, default=None, help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default=None, help='path to a local model state dict to be used')
    parser.add_argument('--alpha', type=int, default=None, help='path to a local model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--image_prep', type=str, default='resize_512x512', help='the image preparation method')
    parser.add_argument('--direction', type=str, default=None, help='the direction of translation. None for pretrained models, a2b or b2a for custom paths.')
    parser.add_argument('--use_fp16', action='store_true', help='Use Float16 precision for faster inference')
    args = parser.parse_args()

    # only one of model_name and model_path should be provided
    if args.model_name is None != args.model_path is None:
        raise ValueError('Either model_name or model_path should be provided')

    if args.model_path is not None and args.prompt is None:
        raise ValueError('prompt is required when loading a custom model_path.')

    if args.model_name is not None:
       # assert args.prompt is None, 'prompt is not required when loading a pretrained model.'
      #  assert args.direction is None, 'direction is not required when loading a pretrained model.'
        pass
    # initialize the model
    alpha = 1
    model = CycleGAN_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()
    if args.use_fp16:
        model.half()
    import glob
    mode = 'inference'
    T_val = build_transform(args.image_prep)
    #input_images = glob.glob('/home/chenghao/GenAI/gaussian_studio/drivestudio/data/waymo/processed/training/023/images_ori/*.jpg')
   # input_images = glob.glob('/media/chenghao/My Passport/Projcet/img2img-turbo/data/day2snow/test_A/*.jpg')
    input_images = glob.glob('/media/chenghao/My Passport/Projcet/img2img-turbo/data/day2snow/test/*.png')

    for input_image_path in  input_images:
            print(input_image_path)
            input_image = Image.open(input_image_path).convert('RGB')
            input_image_label = Image.open(input_image_path.replace('test','train_A_seg').replace('_frame_camera.png','_gt_labelIds.png')).convert("L")

            # translate the image
            with torch.no_grad():

                input_img = T_val(input_image)
                intput_img_label = T_val(input_image_label)
                
                x_t = transforms.ToTensor()(input_img)
                x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()

                x_t_label = transforms.ToTensor()(intput_img_label).unsqueeze(0).cuda()

                if args.use_fp16:
                        x_t = x_t.half()
                output = model(x_t,x_t_label,args.alpha, direction=args.direction, caption=args.prompt,mode=mode)

                output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
                output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)

                # save the output image
                bname = os.path.basename(input_image_path)
                os.makedirs(args.output_dir, exist_ok=True)
                output_pil.save(os.path.join(args.output_dir, bname))
