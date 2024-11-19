import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from cyclegan_turbo_3_weather_bc_processor import CycleGAN_Turbo
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
  #  model.unet.enable_xformers_memory_efficient_attention()
    if args.use_fp16:
        model.half()
    import glob
    mode = 'inference'
    T_val = build_transform(args.image_prep)
    #### Extract image info ######
    image_root = '/home/chenghao/GenAI/gaussian_studio/drivestudio/data/waymo/processed/training/023/images_ori/'
    input_images = glob.glob(image_root+'*.jpg')
   # input_images = glob.glob('/media/chenghao/My Passport/Projcet/img2img-turbo/data/day2snow/test_A/*.jpg')
    frame_list = []
    # input_images = glob.glob('/media/chenghao/My Passport/Projcet/img2img-turbo/data/day2snow/test/*.png')
    for image in input_images:
        frame = image.split('/')[-1].split('_')[0]
        frame_list.append(frame)
    frames = sorted(set(frame_list))
    views = [0,1,2,3,4]
    ###############################
    for frame in frames:
            image_name_0 = frame+'_'+'0.jpg'
            image_name_1 = frame+'_'+'1.jpg'
            image_name_2 = frame+'_'+'2.jpg'
            image_name_3 = frame+'_'+'3.jpg'
            image_name_4 = frame+'_'+'4.jpg'

            input_image_path_0 = os.path.join(image_root,image_name_0)
            input_image_path_1 = os.path.join(image_root,image_name_1)
            input_image_path_2 = os.path.join(image_root,image_name_2)
            input_image_path_3 = os.path.join(image_root,image_name_3)
            input_image_path_4 = os.path.join(image_root,image_name_4)

    
            input_image_0 = Image.open(input_image_path_0).convert('RGB')
            input_image_label_0 = Image.open(input_image_path_0.replace('test','train_A_seg').replace('_frame_camera.png','_gt_labelIds.png')).convert("L")

            input_image_1 = Image.open(input_image_path_1).convert('RGB')
            input_image_label_1 = Image.open(input_image_path_1.replace('test','train_A_seg').replace('_frame_camera.png','_gt_labelIds.png')).convert("L")

            input_image_2 = Image.open(input_image_path_2).convert('RGB')
            input_image_label_2 = Image.open(input_image_path_2.replace('test','train_A_seg').replace('_frame_camera.png','_gt_labelIds.png')).convert("L")

            input_image_3 = Image.open(input_image_path_3).convert('RGB')
            input_image_label_3 = Image.open(input_image_path_3.replace('test','train_A_seg').replace('_frame_camera.png','_gt_labelIds.png')).convert("L")

            input_image_4 = Image.open(input_image_path_4).convert('RGB')
            input_image_label_4 = Image.open(input_image_path_4.replace('test','train_A_seg').replace('_frame_camera.png','_gt_labelIds.png')).convert("L")


            # input_image = torch.stack([input_image_0, input_image_1, input_image_2, input_image_3, input_image_4], dim=1) 
            # input_image_label =  torch.stack([input_image_label_0, input_image_label_1, input_image_label_2, input_image_label_3, input_image_label_4], dim=1) 


            # translate the image
            with torch.no_grad():

                # Apply transformations to the input image and label
                input_img_0 = T_val(input_image_0)
                input_img_label_0 = T_val(input_image_label_0)

                input_img_1 = T_val(input_image_1)
                input_img_label_1 = T_val(input_image_label_1)

                input_img_2 = T_val(input_image_2)
                input_img_label_2 = T_val(input_image_label_2)

                input_img_3 = T_val(input_image_3)
                input_img_label_3 = T_val(input_image_label_3)

                input_img_4 = T_val(input_image_4)
                input_img_label_4 = T_val(input_image_label_4)

                # Convert input image to tensor, normalize, and move to GPU
                x_t_0 = transforms.ToTensor()(input_img_0)
                x_t_0 = transforms.Normalize([0.5], [0.5])(x_t_0).unsqueeze(0)

                x_t_1 = transforms.ToTensor()(input_img_1)
                x_t_1 = transforms.Normalize([0.5], [0.5])(x_t_1).unsqueeze(0)

                x_t_2 = transforms.ToTensor()(input_img_2)
                x_t_2 = transforms.Normalize([0.5], [0.5])(x_t_2).unsqueeze(0)

                x_t_3 = transforms.ToTensor()(input_img_3)
                x_t_3 = transforms.Normalize([0.5], [0.5])(x_t_3).unsqueeze(0)

                x_t_4 = transforms.ToTensor()(input_img_4)
                x_t_4 = transforms.Normalize([0.5], [0.5])(x_t_4).unsqueeze(0)

                # Convert label to tensor and move to GPU
                x_t_label_0 = transforms.ToTensor()(input_img_label_0).unsqueeze(0)
                x_t_label_1 = transforms.ToTensor()(input_img_label_1).unsqueeze(0)
                x_t_label_2 = transforms.ToTensor()(input_img_label_2).unsqueeze(0)
                x_t_label_3 = transforms.ToTensor()(input_img_label_3).unsqueeze(0)
                x_t_label_4 = transforms.ToTensor()(input_img_label_4).unsqueeze(0)

                # Check if GPU is available and move tensors to GPU if possible
                device_0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                x_t_0 = x_t_0.to(device_0)
                x_t_label_0 = x_t_label_0.to(device_0)

                x_t_1 = x_t_1.to(device_0)
                x_t_label_1 = x_t_label_1.to(device_0)

                x_t_2 = x_t_2.to(device_0)
                x_t_label_2 = x_t_label_2.to(device_0)

                x_t_3 = x_t_3.to(device_0)
                x_t_label_3 = x_t_label_3.to(device_0)

                x_t_4 = x_t_4.to(device_0)
                x_t_label_4 = x_t_label_4.to(device_0)

                # Convert to half precision if needed
                if args.use_fp16:
                    x_t_0 = x_t_0.half()
                    x_t_1 = x_t_1.half()
                    x_t_2 = x_t_2.half()
                    x_t_3 = x_t_3.half()
                    x_t_4 = x_t_4.half()

                #print(x_t_0.size())
                x_t = torch.cat([x_t_0, x_t_1, x_t_2, x_t_3, x_t_4], dim=0) 
                x_t_label =  torch.cat([x_t_label_0, x_t_label_1, x_t_label_2, x_t_label_3, x_t_label_4], dim=0) 
                #print(x_t.size(),x_t_label.size())





                output = model(x_t,x_t_label,args.alpha, direction=args.direction, caption=args.prompt,mode=mode)
              #  print(output.size(),'-----------')
                output_pil_0 = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
                output_pil_0 = output_pil_0.resize((input_image_0.width, input_image_0.height), Image.LANCZOS)

                output_pil_1 = transforms.ToPILImage()(output[1].cpu() * 0.5 + 0.5)
                output_pil_1 = output_pil_1.resize((input_image_1.width, input_image_1.height), Image.LANCZOS)

                output_pil_2 = transforms.ToPILImage()(output[2].cpu() * 0.5 + 0.5)
                output_pil_2 = output_pil_2.resize((input_image_2.width, input_image_2.height), Image.LANCZOS)

                output_pil_3 = transforms.ToPILImage()(output[3].cpu() * 0.5 + 0.5)
                output_pil_3 = output_pil_3.resize((input_image_3.width, input_image_3.height), Image.LANCZOS)

                output_pil_4 = transforms.ToPILImage()(output[4].cpu() * 0.5 + 0.5)
                output_pil_4 = output_pil_4.resize((input_image_4.width, input_image_4.height), Image.LANCZOS)



                # save the output image
                os.makedirs(args.output_dir, exist_ok=True)

                bname_0 = os.path.basename(input_image_path_0)
                output_pil_0.save(os.path.join(args.output_dir, bname_0))

                bname_1 = os.path.basename(input_image_path_1)
                output_pil_1.save(os.path.join(args.output_dir, bname_1))

                bname_2 = os.path.basename(input_image_path_2)
                output_pil_2.save(os.path.join(args.output_dir, bname_2))

                bname_3 = os.path.basename(input_image_path_3)
                output_pil_3.save(os.path.join(args.output_dir, bname_3))

                bname_4 = os.path.basename(input_image_path_4)
                output_pil_4.save(os.path.join(args.output_dir, bname_4))