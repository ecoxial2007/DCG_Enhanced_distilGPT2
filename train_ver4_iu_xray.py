import os
import json
import torch
import logging
import argparse
import numpy as np
import random
from torch.cuda.amp import GradScaler
import datetime
from tqdm import tqdm
from dlhpcstarter.utils import load_config_and_update_args
import yaml
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tools.dataset.iu_xray_dataset_forme import Subset
from medsam2distilgpt2_iu_xray import MedSAM2DistilGPT2IUXRay
from tools.metrics.chexbert import CheXbertMetrics
from tools.metrics.coco import COCOCaptionMetrics
seed = 3407

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU

np.random.seed(seed)
random.seed(seed)


# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
current_time = datetime.datetime.now()
timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s", filename=f"./log/log_{timestamp}.txt",
                    filemode='w')
log = logging.getLogger(__name__)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def inplace_move_to_device(data_dict, device):
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.to(device)


def train_one_step(model, batch):
    model.train()
    loss = model(batch['encoder_images'],
                 batch['decoder_input_ids'],
                 batch['decoder_attention_mask'],
                 batch['node_features'],
                 batch['adj_matrix'],
                 batch['label_ids'])
    return loss


def train(model, args, train_dl, test_dl, optimizer, scaler, lr_scheduler):
    test_coco_metrics = COCOCaptionMetrics(metrics=["bleu", "cider", "meteor", "rouge"])
    # CheXbert classification metrics:
    test_chexbert_metrics = CheXbertMetrics(
        bert_path='bert-base-uncased',
        checkpoint_path='stanford/chexbert/chexbert.pth',
        ckpt_dir=args.ckpt_zoo_dir,
        mbatch_size=args.mbatch_size,
        exp_dir=args.exp_dir_trial,
    )
    best_score, best_epoch = 0, 0
    accumulation_steps = 1


    for epoch in range(args.start_epoch, args.epochs):
        pbar = tqdm(train_dl, desc="training begins", total=len(train_dl), ncols=140, leave=True)
        pbar.set_description(f"Epoch: {epoch}")
        model.zero_grad() # Initialize gradients to zero at the start of the epoch

        for steps, batch in enumerate(pbar):

            inplace_move_to_device(batch, args.device)
            loss = train_one_step(model, batch)
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if (steps+1) % accumulation_steps == 0 or steps + 1 == len(pbar):
                scaler.step(optimizer)
                optimizer.zero_grad()
                scaler.update()

            log.info(f"Epoch: {epoch} batch {steps + 1}: loss={loss.item() * accumulation_steps},"
                     f" lr={optimizer.state_dict()['param_groups'][0]['lr']}")
            pbar.set_postfix(steps=steps, loss=loss.data.item() * accumulation_steps,
                             refresh=True)
            # break
        if lr_scheduler is not None:
            lr_scheduler.step()
        pbar.close()
        torch.cuda.empty_cache()

        test_pbar = tqdm(test_dl, desc="test begins", total=len(test_dl), ncols=140, leave=True)
        test_pbar.set_description(f"Test")
        for steps, batch in enumerate(test_pbar):
            inplace_move_to_device(batch, args.device)
            val_one_step(model, batch, test_chexbert_metrics, test_coco_metrics)

        test_scores = validation(test_chexbert_metrics, test_coco_metrics)
        test_metrics = " ".join([f"{k}: {v}" for k, v in test_scores.items()])
        test_pbar.close()
        log.info("Test Results: " + test_metrics)
        monitor = args.monitor[len("val_"):]

        if test_scores[monitor] - best_score > 0.01:
            best_score = test_scores[monitor]
            print(f"best score={best_score}")

            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metric": test_scores,
                "epoch": epoch,
            }
            torch.save(checkpoint, os.path.join(args.work_dir, f"{args.module}_epoch={epoch}_{args.monitor}={test_scores[monitor]}.pth"))
        torch.cuda.empty_cache()


def validation(chexbert_metrics, coco_metrics):
    scores = {}

    output = chexbert_metrics.compute()
    scores.update(output)
    chexbert_metrics.reset()

    output = coco_metrics.compute()
    scores.update(output)
    coco_metrics.reset()
    print({f'val_{k}': v for k, v in scores.items()})
    return scores


def val_one_step(model, batch, chexbert_metrics, coco_metrics, num_beams=1):
    model.eval()
    generated = model.generate(num_beams,
                               batch['encoder_images'],
                               batch['node_features'],
                               batch['adj_matrix'],
                               batch['id'],
                               batch['labels']
                               )
    chexbert_metrics.update(generated, batch['labels'], ids=batch['id'])
    coco_metrics.update(generated, [i for i in batch['labels']], ids=batch['id'])


def main(args):
    args.warm_start_modules = True
    model = MedSAM2DistilGPT2IUXRay(**vars(args))

    if args.resume_path is not None:
        checkpoint = torch.load(args.resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)

    model = model.to(args.device)
    print('finish loading model...')
    scaler = GradScaler()
    # model = DataParallel(model, device_ids=[2, 3])
    with open(model.labels_file_path) as f:
        examples = json.load(f)


    train_dataset = Subset(
        examples=model.format_examples(examples["train"]),
        tokenizer=model.tokenizer,
        decoder_max_len=model.decoder_max_len,
        colour_space='RGB',
        transforms=model.train_transforms,
        self_critical=False,
        train=True,
        add_bos_eos_manually=True,
        num_samples=None)

    test_dataset = Subset(
        examples=model.format_examples(examples["val"]),
        tokenizer=model.tokenizer,
        decoder_max_len=model.decoder_max_len,
        colour_space='RGB',
        transforms=model.test_transforms,
        add_bos_eos_manually=True,
    )
    train_dl = DataLoader(
        train_dataset,
        batch_size=args.mbatch_size,
        num_workers=args.num_workers,
        shuffle=True,
        prefetch_factor=5,
    )

    test_dl = DataLoader(
        test_dataset,
        batch_size=model.mbatch_size,
        num_workers=model.num_workers,
        shuffle=False,
        prefetch_factor=5,
    )
    print('Finish loading dataset..')

    optimizer = model.configure_optimizers()['optimizer']

    if args.resume_path is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    lr_scheduler = MultiStepLR(optimizer, milestones=[8], gamma=0.5)
    train(model, args, train_dl, test_dl, optimizer, scaler, lr_scheduler)



def load_config_from_yaml(path, parser):
    if path.endswith('.yaml'):
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
    for key, value in data.items():
        parser.add_argument(f'--{key}', default=value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/train_iu_xray.yaml")
    parser.add_argument("--epochs", default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--work_dir", default='./experiments/iu_xray')
    parser.add_argument("--trial", default=None)
    parser.add_argument("--fast_dev_run", default=None)
    parser.add_argument("--resume_path", default=None)
    parser.add_argument("--start_epoch", default=0)
    parser.add_argument("--cuda_visible_devices", default=0)
    parser.add_argument("--wd", default=2e-2)
    args = parser.parse_args()
    load_config_and_update_args(args)
    main(args)
