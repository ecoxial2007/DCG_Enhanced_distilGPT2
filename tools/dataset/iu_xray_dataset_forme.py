from PIL import Image
from torch.utils.data import Dataset
import random
import torch
import json
import os, h5py

class Subset(Dataset):
    """
    The base class used to form a subset of the task's dataset. Implemented
    using a torch.utils.data.Dataset for the torch.utils.data.DataLoader for
    the subset of the pytorch_lightning.core.datamodule.LightningDataModule.
    See the tutorial here:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(
            self,
            examples=None,
            transforms=None,
            colour_space=None,
            tokenizer=None,
            encoder_tokenizer=None,
            decoder_max_len=None,
            self_critical=False,
            train=False,
            add_bos_eos_manually=False,
            num_samples=None,
            sample_seed=43,
            **kwargs,
    ):
        """
        Argument/s:
            examples - a list of dictionaries, where each dictionary corresponds to
                an example and has keys that are relevant to the dataset.
            transforms - torchvision transforms to be applied to each image.
            colour_space - color space of the images: "L" (greyscale) or "RGB".
            tokenizer - sentence tokenizer.
            decoder_max_len - maximum length for the decoder's input (training).
            self_critical - self-critical sequence training flag.
            train - training flag.
            add_bos_eos_manually - add the beginning of sentence and end of sentence tokens manually.
            num_samples - subset size of the set.
            sample_seed - seed for the sample.
            kwargs - keyword arguments.
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_max_len = decoder_max_len
        self.transforms = transforms
        self.colour_space = colour_space
        self.self_critical = self_critical
        self.train = train
        self.add_bos_eos_manually = add_bos_eos_manually
        with open('./dataset/iu_x-ray/node_mapping.json', 'r') as f:
            self.node_mapping = json.load(f)

        with h5py.File('./dataset/iu_x-ray/node_features_gpt2.h5', 'r') as h5file:
            self.node_feature = h5file['label_features'][:]

        if num_samples:
            random.seed(sample_seed)
            print(f"Number of examples in dataset: {len(self.examples)}.")
            self.examples = random.sample(self.examples, num_samples)
            print(f"Number of examples in subset of dataset: {len(self.examples)}.")

    def __len__(self):
        return len(self.examples)


    def __getitem__(self, index):
        example = self.examples[index]
        id = example["id"]
        image_1 = self.image_loading_and_preprocessing(example["image_file_path"][0])
        image_2 = self.image_loading_and_preprocessing(example["image_file_path"][1])

        adj_max_path, _ = os.path.split(example["image_file_path"][0])
        adj_max_path = adj_max_path.replace('images', 'adjacency_matrix_191')+'.h5'
        with h5py.File(adj_max_path, 'r') as h5file:
            adj_max = h5file['adj_matrix'][:]
        adj_max = torch.tensor(adj_max, dtype=torch.float32)
        image = torch.stack((image_1, image_2), 0)
        example_dict = {"id": example["id"], "encoder_images": image,
                        "labels": example["label"],
                        "adj_matrix": adj_max,
                        "node_features": self.node_feature}

        if self.train and not self.self_critical:
            example_dict = {**example_dict, **self.tokenize(example["label"])}
        return example_dict


    def tokenize_node(self, node_list):
        # 将所有节点字符串标记化
        tokenized = self.tokenizer(
            node_list,
            padding="max_length",  # 添加填充使所有序列长度相同。
            truncation=True,  # 截断超过最大长度的序列。
            max_length=self.decoder_max_len,  # 设置最大长度。
            return_tensors="pt"  # 返回PyTorch张量。
        )
        example_dict = {
            'node_input_ids': tokenized.input_ids,
            'node_attention_mask': tokenized.attention_mask
        }
        return example_dict

    def image_loading_and_preprocessing(self, image_file_path):
        """
        Load and pre-process an image.

        Argument/s:
            image_file_path - file path to the image.

        Returns:
            image - tensor of the preprocessed image.
        """
        image = Image.open(image_file_path)
        image = image.convert(self.colour_space)  # "L" (greyscale) or "RGB".
        if self.transforms is not None:
            image = self.transforms(image)
        return image

    def tokenize(self, string):
        # 如果需要手动添加开始（BOS）和结束（EOS）标记，则在完整字符串前后添加这些标记。
        if self.add_bos_eos_manually:
            string = self.tokenizer.bos_token + string + self.tokenizer.eos_token

        # 对可能已添加BOS/EOS的完整字符串进行分词，适用于解码器。
        tokenized = self.tokenizer(
            string,
            padding="max_length",  # 添加填充使所有序列长度相同。
            truncation=True,  # 截断超过最大长度的序列。
            max_length=self.decoder_max_len + 1,  # 最大长度设置（+1是因为后面要移除一个token）。
            return_tensors="pt",  # 返回PyTorch张量。
        )

        # 创建一个字典来存储转换后的数据。
        example_dict = {"decoder_input_ids": tokenized.input_ids[0]}
        if "token_type_ids" in tokenized:
            example_dict["decoder_token_type_ids"] = tokenized.token_type_ids[0][1:]


        example_dict["decoder_attention_mask"] = tokenized.attention_mask[0][1:]

        # 标签ID通常是输入ID向右移动一个位置，将前缀和PAD部分都设置为-100。
        labels = example_dict["decoder_input_ids"][1:].detach().clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # 将PAD部分标记为-100
        example_dict["label_ids"] = labels

        # 从解码器输入中移除最后一个token（通常是EOS）。
        example_dict["decoder_input_ids"] = example_dict["decoder_input_ids"][:-1]

        return example_dict