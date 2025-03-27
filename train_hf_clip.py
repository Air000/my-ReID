import logging
import time
import torch
from transformers import CLIPProcessor, CLIPModel
import os
import os.path as osp
import glob
import json
import re
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (input, pid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size=16, num_instances=4):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        #for index, (_, pid, _, _) in enumerate(self.data_source):
        for index, (_, pid, _, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length

# ================================
# 1. Dataset
# ================================
class T2IDataset(Dataset):
    """
    dataset: a list of tuples: (img_path, pid, camid, viewid, caption)
    processor: a CLIPProcessor for image & text processing
    """
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, cid, vid, cap = self.dataset[index]
        img = Image.open(img_path).convert("RGB")

        inputs = self.processor(text=cap, images=img, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        
        # Remove the batch dimension from each tensor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        

        return inputs, torch.tensor(pid, dtype=torch.int64), img_path

class CustomCLIPDataset(object):
    dataset_dir = 'VeRi/'

    def __init__(self, root='', verbose=True, **kwargs):
        super(CustomCLIPDataset, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')
        self.split_path = osp.join(self.dataset_dir, 'reid.json')

        self._check_before_run()

        # Process the splits
        train, query, gallery = self._process_split()

        self.train = train
        self.query = query
        self.gallery = gallery

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not osp.exists(self.split_path):
            raise RuntimeError("'{}' is not available".format(self.split_path))

    def _process_split(self):
        with open(self.split_path, 'r') as f:
            data = json.load(f)

        train_set = []
        test_set = []

        for item in data:
            img_path = osp.join(self.dataset_dir, item['file_path'])
            pid = item['id']
            split = item['split']
            captions = item['captions'][0]
            camid = int(re.search(r'_c(\d+)', item['file_path']).group(1))
            camid -= 1  # index starts from 0

            if split == 'train':
                train_set.append((img_path, pid, camid, 0, captions))  # viewid is set to 0 for simplicity
            elif split == 'test':
                test_set.append((img_path, pid, camid, 0, captions))

            # Split test set into query and gallery
            query_set = []
            gallery_set = []
            pid_to_images = {}

            # Group images by pid
            for img_path, pid, camid, viewid, captions in test_set:
                if pid not in pid_to_images:
                    pid_to_images[pid] = []
                pid_to_images[pid].append((img_path, camid, viewid, captions))

            # For each pid, randomly assign one image to the query set and the rest to the gallery set
            for pid, images in pid_to_images.items():
                query_set.append((images[0][0], pid, images[0][1], images[0][2], images[0][3]))
                for img in images[1:]:
                    gallery_set.append((img[0], pid, img[1], img[2], img[3]))

        return train_set, query_set, gallery_set
    
# ================================
# 2. Define Batch-Hard Triplet Loss
# ================================
def batch_hard_triplet_loss(embeddings, labels, margin=0.3):
    """
    Computes the batch-hard triplet loss.
    
    embeddings: Tensor of shape (batch_size, embedding_dim)
    labels: Tensor of shape (batch_size)
    margin: Triplet loss margin
    """
    # Compute pairwise Euclidean distances
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)  # shape: (batch_size, batch_size)
    
    loss = torch.tensor(0.0, device=embeddings.device)
    batch_size = embeddings.size(0)
    for i in range(batch_size):
        # Positive samples (same label) but not the anchor itself
        pos_mask = (labels == labels[i]) & (torch.arange(batch_size, device=labels.device) != i)
        # Negative samples (different label)
        neg_mask = labels != labels[i]
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue
        hardest_positive = pairwise_dist[i][pos_mask].max()
        hardest_negative = pairwise_dist[i][neg_mask].min()
        loss += F.relu(hardest_positive - hardest_negative + margin)
    return loss / batch_size

# AverageMeter for tracking metrics
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0
        self.count = 0
    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0


# Set up logger (or use your existing logger)
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Load pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

dataset = CustomCLIPDataset(root="../")
dataset_train = T2IDataset(dataset.train, processor)
dataset_query = T2IDataset(dataset.query, processor)
dataset_gallery = T2IDataset(dataset.gallery, processor)
train_loader = DataLoader(dataset_train, batch_size=64, sampler=RandomIdentitySampler(dataset.train), num_workers=2)
query_loader = DataLoader(dataset_query, batch_size=64, shuffle=False, num_workers=2)
gallery_loader = DataLoader(dataset_gallery, batch_size=64, shuffle=False, num_workers=2)

loss_meter = AverageMeter()
acc_meter = AverageMeter()

log_period = 20  # log every 10 iterations
start_time = time.time()

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)

# Training loop
num_epochs = 101
for epoch in range(num_epochs):
    model.train()
    loss_meter.reset()
    acc_meter.reset()

    for n_iter, (inputs, labels, _) in enumerate(train_loader):
        # batch is a tuple: (inputs, label)
        labels = labels.to(device)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        optimizer.zero_grad()

        text_features = model.get_text_features(inputs['input_ids'])
        image_features = model.get_image_features(inputs['pixel_values'])

        embeddings = (text_features + image_features) / 2  # shape: (batch_embedding_dim)

        loss = batch_hard_triplet_loss(embeddings, labels, margin=0.3)
        # print(loss)
        loss.backward()
        optimizer.step()

        logits = image_features @ text_features.T
        batch_acc = (logits.argmax(dim=1) == labels).float().mean()

        # Update meters (use the batch size; here assuming inputs['pixel_values'].size(0) is batch size)
        batch_size = inputs['pixel_values'].size(0)
        loss_meter.update(loss.item(), n=batch_size)
        acc_meter.update(batch_acc.item(), n=batch_size)

        # Ensure all CUDA operations are finished
        torch.cuda.synchronize()

        if (n_iter + 1) % log_period == 0:
            logger.info(
                "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}"
                .format(
                    epoch,
                    (n_iter + 1),
                    len(train_loader),
                    loss_meter.avg,
                    acc_meter.avg
                )
            )
    end_time = time.time()
    time_per_batch = (end_time - start_time) / (n_iter + 1)
    logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

    if epoch % 20 == 0:
        # Save fine-tuned model
        model.save_pretrained("fine_tuned_clip_model_{}".format(epoch))
        processor.save_pretrained("fine_tuned_clip_processor_{}".format(epoch))
