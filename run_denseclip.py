import torch

from single_denseclip.dense_clip import DenseCLIP
from utils.utils import get_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# directory configurations - change the below three lines as appropriate
#dir_ckpt: "/users/gyungin/reco/ckpt"  # Change this to your checkpoint directory.
dir_dataset="/home/myb/datasets/cityscapes"  # Change this to your dataset directory.

dataset_name="cityscapes"  # ["cityscapes", "coco_stuff", "kitti_step", "pascal_context"]

# hyperparameters for networks
clip_arch="ViT-L/14@336px"  # For image retrieval. ["RN50", "RN50x16", "RN50x64", "ViT-L/14@336px"]
dense_clip_arch="RN50x16"  # ["RN50", "RN50x16", "RN50x64"]
#dense_clip_inference=true
encoder_arch: "DeiT-S/16-SIN"  # For an image encoder. ["RN50", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "mocov2", "swav", "dino_small", "dino_base", "DeiT-S/16-SIN"]
patch_size: 16

# hyperparameters for ReCo/ReCo+ framework
batch_size: 16
context_categories=["tree", "sky", "building", "road", "person"]
#context_elimination: true

n_imgs: 50  # the number of images of the same category
n_workers=16
#text_attention: true


dataset, categories, palette = get_dataset(
        dir_dataset=dir_dataset,
        dataset_name=dataset_name,
        split="train",
        dense_clip_arch=dense_clip_arch
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DenseCLIP(arch_name="RN50x16").to(device)
#print(model.state_dict)

dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        num_workers=n_workers,
        pin_memory=True
    )

#iter()函数生成迭代器
iter_dataloader, pbar = iter(dataloader), tqdm(range(len(dataloader)))
print("len(dataloader):",len(dataloader))
for num_batch in pbar:
    dict_data = next(iter_dataloader)
    print("type(dict_data):",type(dict_data))
    val_img: torch.Tensor = dict_data["img"]  # b x 3 x H x W
    print("val_img",val_img.shape)
    #val_gt: torch.LongTensor = dict_data["gt"]  # b x H x W

    dt: torch.Tensor = model(val_img.to(device))  # b x n_cats x H x W
    print("dt_img",dt.shape)
    #val_img torch.Size([16, 3, 320, 320]) batchsize:16, RGB:3, H:320,W:320
    #dt_img torch.Size([16, 768, 10, 10])
