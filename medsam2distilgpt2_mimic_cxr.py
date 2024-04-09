import os
import contextlib
import torch
import torch.nn.functional as F
import transformers
from lightning.pytorch import LightningModule
import numpy as np
from torchvision import transforms
from transformers.configuration_utils import PretrainedConfig
from tools.graph import Graph
from tools.dataset.mimic_cxr_chen_tokenizer import TokenizerChen
from tools.new_module import ResidualCrossAttentionBlock
from tools.multi_image import MultiImageInput, MultiImageOutput
from segment_anything import sam_model_registry
from tools.metrics.chexbert import CheXbertMetrics
from tools.metrics.coco import COCOCaptionMetrics
from tools.metrics.report_logger import ReportLogger
from PIL import Image

class Convert2Dto3D:
    def __call__(self, img):
        if img.mode == 'L':  # 'L' mode indicates a single-channel image
            img = np.array(img)
            img = np.repeat(img[:, :, None], 3, axis=-1)
            img = Image.fromarray(img)
        return img

class MinMaxNormalization:
    def __call__(self, img):
        img = np.array(img, dtype=np.float32)
        img = (img - img.min()) / np.clip(img.max() - img.min(), a_min=1e-8, a_max=None)
        return img

class MedSAM2DistilGPT2MIMICXR(LightningModule):
    def __init__(
            self,
            warm_start_modules: bool,
            exp_dir_trial: str,
            dataset_dir: str,
            ckpt_zoo_dir: str,
            mbatch_size: int,
            encoder_lr: float,
            decoder_lr: float,
            decoder_max_len: int,
            num_test_beams: int,
            prefetch_factor: int = 5,
            num_workers: int = 0,
            wd: float = 2e-2,
            **kwargs,
    ):
        super().__init__()

        self.warm_start_modules = warm_start_modules
        self.exp_dir_trial = exp_dir_trial
        self.dataset_dir = dataset_dir
        self.ckpt_zoo_dir = ckpt_zoo_dir
        self.mbatch_size = mbatch_size
        self.encoder_lr = encoder_lr
        self.decoder_lr = decoder_lr
        self.decoder_max_len = decoder_max_len
        self.num_test_beams = num_test_beams
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.wd = wd

        # Paths:
        self.labels_file_path = os.path.join(self.dataset_dir, "mimic_cxr", "annotation_top5.json")
        self.dataset_dir = os.path.join(self.dataset_dir, "mimic_cxr", "images")
        self.chen_tokenizer = TokenizerChen(
            ann_path=self.labels_file_path,
            threshold=3,
        )
        self.chen_max_seq_length = 60

        """
        Evaluation metrics
        
        These need to be defined correctly in order for them to be placed on the correct device:
        https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html#torchmetrics-in-pytorch-lightning
        """
        self.val_coco_metrics = COCOCaptionMetrics(metrics=["bleu", "cider", "rouge"])
        self.test_coco_metrics = COCOCaptionMetrics(metrics=["bleu", "cider", "meteor", "rouge"])

        # CheXbert classification metrics:
        self.val_chexbert_metrics = CheXbertMetrics(
            bert_path='bert-base-uncased',
            checkpoint_path='stanford/chexbert/chexbert.pth',
            ckpt_dir=self.ckpt_zoo_dir,
            mbatch_size=self.mbatch_size,
            exp_dir=self.exp_dir_trial,
        )
        self.test_chexbert_metrics = CheXbertMetrics(
            bert_path='bert-base-uncased',
            checkpoint_path='stanford/chexbert/chexbert.pth',
            ckpt_dir=self.ckpt_zoo_dir,
            mbatch_size=self.mbatch_size,
            exp_dir=self.exp_dir_trial,
        )

        # Report logging:
        self.val_report_logger = ReportLogger(exp_dir=self.exp_dir_trial, split='val_reports')
        self.test_report_logger = ReportLogger(exp_dir=self.exp_dir_trial, split='test_reports')


        MedSAM_CKPT_PATH = './checkpoint/MedSAM/medsam_vit_b.pth'
        self.sam_encoder = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
        # for param in self.sam_encoder.parameters():
        #     param.requires_grad = False

        self.cross_attn_g2i = ResidualCrossAttentionBlock(d_model=768, n_head=12, dropout=0.1)
        self.cross_attn_i2g = ResidualCrossAttentionBlock(d_model=768, n_head=12, dropout=0.1)


        self.graph = Graph(dim_in=768, dim_hidden=128, dim_out=768, num_layers=2, dropout=0.1)
        self.multi_input = MultiImageInput()
        self.multi_output = MultiImageOutput()

        # Decoder:
        ckpt_name = 'distilgpt2'
        config = transformers.GPT2Config.from_pretrained(
            os.path.join(self.ckpt_zoo_dir, ckpt_name),
            local_files_only=True,
        )
        config.add_cross_attention = True
        config.is_decoder = True

        if self.warm_start_modules:
            decoder = transformers.GPT2LMHeadModel.from_pretrained(
                os.path.join(self.ckpt_zoo_dir, ckpt_name),
                local_files_only=True,
                config=config,
            )
        else:
            decoder = transformers.GPT2LMHeadModel(config=config)


        # Resize GPT2 embedding to include padding and beginning of sentence token:
        decoder.resize_token_embeddings(config.vocab_size + 3)

        # Decoder tokenizer:
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained(
            os.path.join(self.ckpt_zoo_dir, ckpt_name),
            local_files_only=True,
        )
        self.tokenizer.add_special_tokens({"bos_token": "[BOS]", 'pad_token': '[PAD]', "cls_token": "[CLS]"})


        # Print the special tokens:
        print('Description, Special token, Index')
        for k, v in self.tokenizer.special_tokens_map.items():
            if k != 'additional_special_tokens':
                print(f'{k}, {v}, {getattr(self.tokenizer, k + "_id")}')
            else:
                for i, j in zip(self.tokenizer.additional_special_tokens, self.tokenizer.additional_special_tokens_ids):
                    print(f'additional_special_token, {i}, {j}')

        # We don't actually want to use the encoder of the EncoderDecoderModel, create a dummy encoder:
        class DummyEncoder:
            main_input_name = 'dummy'

            class DummyConfig(PretrainedConfig):
                model_type = 'bert'

            config = DummyConfig()

            def __init__(self, hidden_size):
                self.config.hidden_size = hidden_size

            def get_output_embeddings(cls):
                return None

        # Use Hugging Face Transformers EncoderDecoderModel to generate conditionally:
        dummy_encoder = DummyEncoder(hidden_size=decoder.config.hidden_size)

        # To be compatible with previous the framework (and hence, the available checkpoint):
        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_decoder = transformers.EncoderDecoderModel(encoder=dummy_encoder, decoder=decoder)

        self.decoder = Decoder()

        # Image transformations:
        self.train_transforms = transforms.Compose(
            [
                Convert2Dto3D(),
                transforms.Resize((512, 512)),
                MinMaxNormalization(),
                transforms.ToTensor(),

            ]
        )
        self.test_transforms = transforms.Compose(
            [
                Convert2Dto3D(),
                transforms.Resize((512, 512)),
                MinMaxNormalization(),
                transforms.ToTensor(),

            ]
        )



    def format_examples(self, examples):
        for i in examples:
            i["image_file_path"] = i.pop("image_path")
            i["label"] = i.pop("report")
            i["image_file_path"] = [j for j in i["image_file_path"]]
            i["label"] = self.chen_tokenizer(i["label"])[:self.chen_max_seq_length]
            i["label"] = self.chen_tokenizer.decode(i["label"][1:])

        return examples


    def configure_optimizers(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """
        grouped_parameters = [
            {"params": self.sam_encoder.parameters(), 'lr': self.encoder_lr},
            {"params": self.decoder.parameters(), 'lr': self.decoder_lr},
        ]

        optimiser = {'optimizer': torch.optim.AdamW(grouped_parameters, lr=self.decoder_lr, weight_decay=self.wd)}
        return optimiser

    def encoder_forward(self, images, adj_matrix=None, node_features=None):
        """
        Encoder forward propagation.

        Argument/s:
            images - a mini-batch of images.
            image_batch_ids - batch index for each image.

        Returns:
            encoder_outputs - transformers.modeling_outputs.ModelOutput.
        """
        B, V, _, _, _ = images.shape
        views = self.multi_input(images)


        image_features = self.sam_encoder.image_encoder(views['images'])
        image_features = image_features.view(*image_features.shape[:2], -1).permute(0, 2, 1)
        image_features = self.multi_output(image_features, views['images_per_example'])['last_hidden_state'].contiguous()

        if adj_matrix is not None:
            node_features = self.graph(node_features, adj_matrix).permute(1, 0, 2)
            image_features_i2g = self.cross_attn_i2g(image_features.permute(1, 0, 2), node_features, node_features).permute(1, 0, 2)
            node_features_g2i = self.cross_attn_g2i(node_features, image_features.permute(1, 0, 2), image_features.permute(1, 0, 2)).permute(1, 0, 2)
            image_features = torch.cat([node_features_g2i, image_features_i2g], dim=1)

        encoder_outputs = transformers.modeling_outputs.BaseModelOutput(last_hidden_state=image_features)
        return encoder_outputs

    def forward(self, images, decoder_input_ids, decoder_attention_mask,
                              node_features, adj_matrix,
                              label_ids):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#forward
        """
        with self.maybe_autocast():
            encoder_outputs = self.encoder_forward(images, adj_matrix, node_features)

            outputs = self.decoder.encoder_decoder(
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                return_dict=True,
            )

            loss = F.cross_entropy(
                outputs.logits.permute([0, 2, 1]), label_ids, ignore_index=-100,
            )
            return loss

    def generate(self, num_beams, images,
                 node_features, adj_matrix, name, report):
        """
        Autoregressively generate a prediction.

        Argument/s:
            num_beams - number of considered beams for the search (one beam is a greedy search).
            images - images for the encoder.

        Returns:
            Indices of the tokens for the predicted sequence.
        """

        with self.maybe_autocast():
            encoder_outputs = self.encoder_forward(images, adj_matrix, node_features)
            outputs = self.decoder.encoder_decoder.generate(
                special_token_ids=[self.tokenizer.sep_token_id],
                max_length=self.decoder_max_len,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                mask_token_id=self.tokenizer.pad_token_id,
                num_beams=num_beams,
                return_dict_in_generate=True,
                use_cache=True,
                encoder_outputs=encoder_outputs,
            )
            """
                generate() for transformers==4.35.2
            inputs_embeds = torch.randn([images.shape[0], 20, 1024])
            outputs = self.decoder.encoder_decoder.generate(
                inputs_embeds=inputs_embeds,
                max_length=self.decoder_max_len,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=num_beams,
                return_dict_in_generate=True,
                use_cache=True,
                encoder_outputs=encoder_outputs,
            )"""
            captions = self.tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)
        return captions

    def maybe_autocast(self, dtype=torch.float16):
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()


