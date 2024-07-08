import random
from collections import defaultdict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

class FewShotBatchSampler:
    def __init__(self, speaker_id, N_way, K_shot, include_query=False, shuffle=True, shuffle_once=False):
        """
        Inputs:
            dataset_targets - speaker ids of training speaker
            N_way - Number of classes to sample per batch.
            K_shot - Number of examples to sample per class in the batch.
            include_query - If True, returns batch of size N_way*K_shot*2, which
                            can be split into support and query set. Simplifies
                            the implementation of sampling the same classes but
                            distinct examples for support and query set.
            shuffle - If True, examples and classes are newly shuffled in each
                      iteration (for training)
            shuffle_once - If True, examples and classes are shuffled once in
                           the beginning, but kept constant across iterations
                           (for validation)
        """
        super().__init__()
        self.dataset_targets = speaker_id
        self.N_way = N_way
        self.K_shot = K_shot
        self.shuffle = shuffle
        self.include_query = include_query
        if self.include_query:
            self.K_shot *= 2
        self.batch_size = self.N_way * self.K_shot  # Number of overall images per batch

        # Organize examples by class
        self.classes = torch.unique(self.dataset_targets).tolist()
        self.num_classes = len(self.classes)
        self.indices_per_class = {}
        self.batches_per_class = {}  # Number of K-shot batches that each class can provide
        for c in self.classes:
            self.indices_per_class[c] = torch.where(self.dataset_targets == c)[0]
            self.batches_per_class[c] = self.indices_per_class[c].shape[0] // self.K_shot

        # Create a list of classes from which we select the N classes per batch
        self.iterations = sum(self.batches_per_class.values()) // self.N_way
        self.class_list = [c for c in self.classes for _ in range(self.batches_per_class[c])]
        if shuffle_once or self.shuffle:
            self.shuffle_data()
        else:
            # For testing, we iterate over classes instead of shuffling them
            sort_idxs = [
                i + p * self.num_classes for i, c in enumerate(self.classes) for p in range(self.batches_per_class[c])
            ]
            self.class_list = np.array(self.class_list)[np.argsort(sort_idxs)].tolist()

    def shuffle_data(self):
        # Shuffle the examples per class
        for c in self.classes:
            perm = torch.randperm(self.indices_per_class[c].shape[0])
            self.indices_per_class[c] = self.indices_per_class[c][perm]
        # Shuffle the class list from which we sample. Note that this way of shuffling
        # does not prevent to choose the same class twice in a batch. However, for
        # training and validation, this is not a problem.
        random.shuffle(self.class_list)

    def __iter__(self):
        # Shuffle data
        if self.shuffle:
            self.shuffle_data()

        # Sample few-shot batches
        start_index = defaultdict(int)
        for it in range(self.iterations):
            class_batch = self.class_list[it * self.N_way: (it + 1) * self.N_way]  # Select N classes for the batch
            index_batch = []
            for c in class_batch:  # For each class, select the next K examples and add them to the batch
                index_batch.extend(self.indices_per_class[c][start_index[c]: start_index[c] + self.K_shot])
                start_index[c] += self.K_shot
            if self.include_query:  # If we return support+query set, sort them so that they are easy to split
                index_batch = index_batch[::2] + index_batch[1::2]
            yield index_batch

    def __len__(self):
        return self.iterations


class TaskBatchSampler:
    def __init__(self, dataset_targets, batch_size, N_way, K_shot, include_query=False, shuffle=True):
        """
        Inputs:
            dataset_targets - PyTorch tensor of the labels of the data elements.
            batch_size - Number of tasks to aggregate in a batch
            N_way - Number of classes to sample per batch.
            K_shot - Number of examples to sample per class in the batch.
            include_query - If True, returns batch of size N_way*K_shot*2, which
                            can be split into support and query set. Simplifies
                            the implementation of sampling the same classes but
                            distinct examples for support and query set.
            shuffle - If True, examples and classes are newly shuffled in each
                      iteration (for training)
        """
        super().__init__()
        self.batch_sampler = FewShotBatchSampler(dataset_targets, N_way, K_shot, include_query, shuffle)
        self.task_batch_size = batch_size
        self.local_batch_size = self.batch_sampler.batch_size

    def __iter__(self):
        # Aggregate multiple batches before returning the indices
        batch_list = []
        for batch_idx, batch in enumerate(self.batch_sampler):
            batch_list.extend(batch)
            if (batch_idx + 1) % self.task_batch_size == 0:
                yield batch_list
                batch_list = []

    def __len__(self):
        return len(self.batch_sampler) // self.task_batch_size

    def get_collate_fn(self):
        def collate_fn(item_list):
            audio_batch, vertice_batch, template_batch, speaker_batch = [], [], [], []

            for batch in item_list:
                audio, vert, temp, speaker,_ = batch
                audio_batch.append(audio)
                vertice_batch.append(vert)
                template_batch.append(temp)
                speaker_batch.append(speaker)
            vertice_mask_batch = [torch.ones(each.shape[0]) for each in vertice_batch]

            audio_batch = pad_sequence(audio_batch, batch_first=True).float()
            vertice_batch = pad_sequence(vertice_batch, batch_first=True).float()
            vertice_mask_batch = pad_sequence(vertice_mask_batch, batch_first=True).float()
            template_batch = pad_sequence(template_batch, batch_first=True).float()
            speaker_batch=torch.tensor(speaker_batch).float()
            audio_batch=audio_batch.chunk(self.task_batch_size,dim=0)
            vertice_mask_batch=vertice_mask_batch.chunk(self.task_batch_size,dim=0)
            vertice_batch=vertice_batch.chunk(self.task_batch_size,dim=0)
            template_batch=template_batch.chunk(self.task_batch_size,dim=0)
            speaker_batch=speaker_batch.chunk(self.task_batch_size,dim=0)

            return list(zip(audio_batch, vertice_batch, vertice_mask_batch, template_batch,speaker_batch))
        return collate_fn

