import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from cyclegan_turbo_3_weather_bc_processor import CycleGAN_Turbo
from my_utils.training_utils_3_weather import build_transform
import glob
from collections import deque


def load_and_prepare_image(image_path, transform_fn):
    image = Image.open(image_path).convert('RGB')
    label_path = image_path.replace('test', 'test_A_seg').replace('_frame_camera.png', '_gt_labelIds.png')
   # label_path = image_path.replace('images', 'mask').replace('.jpg', '_gt_labelIds.jpg')
    print(label_path)
    label = Image.open(label_path).convert('L')
    print(label_path,'-----')
    return transform_fn(image), transform_fn(label)


# def load_and_prepare_images(image_paths, transform_fn):
#     """
#     加载5张图片并拼接为一个张量，同时加载标签并进行拼接。
#     """
#     images = []
#     labels = []
#     for image_path in image_paths:
#         image = Image.open(image_path).convert('RGB')
#         label_path = image_path.replace('images', 'mask').replace('.jpg', '_gt_labelIds.jpg')
#         label = Image.open(label_path).convert('L')
        
#         # 进行预处理
#         images.append(transform_fn(image))
#         labels.append(transform_fn(label))
    
#     # 将5张图片和标签拼接在一起
#     images_concat = torch.cat([transforms.ToTensor()(img).unsqueeze(0) for img in images], dim=0)  # 按宽度拼接
#     labels_concat = torch.cat([transforms.ToTensor()(label).unsqueeze(0) for label in labels], dim=0)  # 按宽度拼接
#     return images_concat, labels_concat



def load_in_batches_with_deque(arr, batch_size=2):
    result = []
    window = deque(maxlen=batch_size)  # 限定 deque 大小，形成滑动窗口

    for item in arr:
        window.append(item)  # 将当前元素添加到 deque
        if len(window) == batch_size:
            # 将当前批次加入结果，复制 deque 内容生成列表
            result.append(list(window))

    return result

def tensorize_image(image, normalize=True):
    tensor = transforms.ToTensor()(image)
    if normalize:
        tensor = transforms.Normalize([0.5], [0.5])(tensor)
    return tensor.unsqueeze(0)


def save_output_images(output_tensors, input_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
 #   print(output_tensors.size(),'1111')
    for i, output_tensor in enumerate(output_tensors):
      #  print(i)
        output_pil = transforms.ToPILImage()(output_tensor.cpu() * 0.5 + 0.5)
        if i < 5:
            output_pil = output_pil.resize((Image.open(input_paths[i]).width, Image.open(input_paths[i]).height), Image.LANCZOS)
            output_pil.save(os.path.join(output_dir, os.path.basename(input_paths[i])))
        else:
            i = i%5
            output_pil = output_pil.resize((Image.open(input_paths[i]).width, Image.open(input_image_paths_next[i]).height), Image.LANCZOS)
            output_pil.save(os.path.join(output_dir, os.path.basename(input_image_paths_next[i])))         


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

    # Ensure either model_name or model_path is provided, not both or neither
    if args.model_name is None == args.model_path is None:
        raise ValueError('Either model_name or model_path should be provided')
    if args.model_path is not None and args.prompt is None:
        raise ValueError('Prompt is required when loading a custom model_path.')

    # Initialize the model
    model = CycleGAN_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.eval()
    if args.use_fp16:
        model.half()

    T_val = build_transform(args.image_prep)
   # image_root = '/home/chenghao/GenAI/gaussian_studio/drivestudio/data/waymo/processed/training/023/images/'
   # image_root = '/media/chenghao/My Passport/Projcet/img2img-turbo/data/day2snow/test_A/'
    image_root = '/media/chenghao/My Passport/Projcet/drivestudio/data/pandaset/processed/044/images/'
    input_images = glob.glob(image_root + '*.jpg')
    frames = sorted(set(image.split('/')[-1].split('_')[0] for image in input_images))
   # print(frames)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batches = load_in_batches_with_deque(frames)
    for i, batch in enumerate(batches):
        frame,frame_next = batch
    # for frame in frames:
        # frame_next = f"{int(frame) + 1:03d}"
        input_image_paths = [os.path.join(image_root, f'{frame}_{view}.jpg') for view in range(5)]
        images, labels = zip(*(load_and_prepare_image(path, T_val) for path in input_image_paths))


        input_image_paths_next = [os.path.join(image_root, f'{frame_next}_{view}.jpg') for view in range(5)]
        images_next, labels_next = zip(*(load_and_prepare_image(path, T_val) for path in input_image_paths_next))

        # Convert images and labels to tensors
        x_t = torch.cat([tensorize_image(img).to(device) for img in images], dim=0)
        x_t_label = torch.cat([tensorize_image(label, normalize=False).to(device) for label in labels], dim=0)


        x_t_next = torch.cat([tensorize_image(img).to(device) for img in images_next], dim=0)
        x_t_label_next = torch.cat([tensorize_image(label, normalize=False).to(device) for label in labels_next], dim=0)

        x_t = torch.cat([x_t,x_t_next],dim=0)
        x_t_label = torch.cat([x_t_label,x_t_label_next],dim=0)
        print(x_t.size(),x_t_label.size())
        # Convert to half precision if needed
        if args.use_fp16:
            x_t = x_t.half()

        # Perform inference
        with torch.no_grad():
            output = model(x_t, x_t_label, args.alpha, direction=args.direction, caption=args.prompt, mode='inference')

        # Save the output images
        save_output_images(output, input_image_paths, args.output_dir)
