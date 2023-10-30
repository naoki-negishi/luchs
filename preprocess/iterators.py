import json
from itertools import islice
from pathlib import Path
from typing import Optional, Union

import torch
from logzero import logger
from luchs.preprocess.instances import (ENC_labels, HolCCGBatchInstance,
                                        SNLIBatchInstance, SNLIEvalInfo,
                                        ModelTypes)
from tqdm import tqdm

BatchInstance = Union[SNLIBatchInstance, HolCCGBatchInstance]


class DataLoader:
    def __init__(
        self,
        file_path: str,
        batch_size: int,
        data_size_percentage: float,
        shuffle: bool,
        model_type: str,
        dataset: Optional[list[BatchInstance]] = None,
    ):
        self.file_path = file_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        assert model_type in ModelTypes

        if self.dataset:
            self.dataset = dataset
        else:
            self.load_dataset(self.file_path)

        if self.model_type == "snli":
            self.batches = self.create_snli_batches()
        elif self.model_type == "holccg":
            self.batches = self.create_holccg_batches()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def __iter__(self):
        return self

    def __next__(self):
        # if self.batch_index >= len(self.batches):
        #     raise StopIteration
        # batch = self.batches[self.batch_index]
        # self.batch_index += 1
        # return batch
        return self.padding(next(self.batches))

    def __len__(self):
        return len(self.batches)

    def load_dataset(self, file_path: str) -> list[dict]:
        self.dataset: list[dict] = []

        assert Path(self.file_path).exists(), f"not found: {self.file_path}"
        logger.info(f"Loading file: '{self.file_path}'")
        len_file = sum(1 for _ in open(self.file_path))
        logger.info(f"The number of lines: {len_file}")
        logger.info(f"The percentage of using data: {self.data_size_percentage} %")

        with open(self.file_path) as f:
            for jsonl in tqdm(
                islice(
                    f,
                    int(len_file * self.data_size_percentage / 100),
                )
            ):
                instance = json.loads(jsonl)
                # if instance["xs_len"] <= self.max_seq_length:
                self.dataset.append(instance)

        logger.info(f"The number of instances: {len(self.dataset)}")
        # logger.info(f"The number of filtered instances: {len_file - len(self.dataset)}")
        logger.info(f"The number of batches: {len(self)}")

    def create_snli_batches(self) -> list[SNLIBatchInstance]:
        def padding(batch: list[dict]) -> SNLIBatchInstance:
            premise: list[str] = [instace["sentece1"] for instace in batch]
            hypothesis: list[str] = [instace["sentece2"] for instace in batch]
            gold_label: list[int] = torch.LongTensor(
                [ENC_labels[instace["gold_label"]] for instace in batch]
            )
            EvalInfo: list[dict] = [
                SNLIEvalInfo(
                    annotator_labels=instace["annotator_labels"],
                    captionID=instace["captionID"],
                    pairID=instace["pairID"],
                    sentence1_binary_parse=instace["sentence1_binary_parse"],
                    sentence1_parse=instace["sentence1_parse"],
                    sentence2_binary_parse=instace["sentence2_binary_parse"],
                    sentence2_parse=instace["sentence2_parse"],
                )
                for instace in batch
            ]

            return SNLIBatchInstance(
                premise=premise,
                hypothesis=hypothesis,
                gold_label=gold_label,
                EvalInfo=EvalInfo,
            )

        minibatch: list[SNLIBatchInstance] = []
        for ex in self.dataset:
            minibatch.append(ex)
            if len(minibatch) == self.batch_size:
                minibatch_instance = padding(minibatch)
                yield minibatch_instance
                minibatch = []
            elif len(minibatch) > self.batch_size:
                raise ValueError
                # yield minibatch[:-1]
                # minibatch = minibatch[-1:]
        if minibatch:
            minibatch_instance = padding(minibatch)
            yield minibatch_instance

    def create_holccg_batches(self, file_path: str) -> list[HolCCGBatchInstance]:
        raise NotImplementedError
