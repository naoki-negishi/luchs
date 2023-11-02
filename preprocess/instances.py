import torch
from typing import TypedDict

ENC_labels = {"entailment": 0, "neutral": 1, "contradiction": 2, "-": 3}

BASECLS = "Baseline-CLS"
BASEBIENCODER = "Baseline-BiEncode"
HOLCCG = "HolCCG"
ModelTypes = [BASECLS, BASEBIENCODER, HOLCCG]


class SNLIEvalInfo(TypedDict):
    annotator_labels: list[str]
    captionID: str
    pairID: str
    sentence1_binary_parse: str
    sentence1_parse: str
    sentence2_binary_parse: str
    sentence2_parse: str


class SNLIBatchInstance(TypedDict):
    premise: list[str]
    hypothesis: list[str]
    gold_label: torch.LongTensor
    EvalInfo: list[SNLIEvalInfo]


class HolCCGBatchInstance(TypedDict):
    pass
