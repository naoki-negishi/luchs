import argparse
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

# import hydra
import logzero
import numpy as np
import torch
import torch.optim as optim
import wandb
import yaml
from logzero import logger
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.models import BaseBiEncoderModel, BaseCLSModel
from preprocess.instances import (BASEBIENCODER, BASECLS, HOLCCG, ENC_labels,
                                  ModelTypes)
from preprocess.iterators import MyDataLoader
from torch import nn
from tqdm import tqdm
from transformers import (RobertaConfig, RobertaModel,
                          get_cosine_schedule_with_warmup)



LOG_FILE_BASENAME = f"{datetime.now().strftime('%Y%m%d-%H%M')}.log"
RoBERTaBase = "roberta-base"


@dataclass
class FinetuningArgs:
    """The parameteers will be loaded from a yaml file"""

    # load and save
    train_data_file: str
    dev_data_file: str
    output_dir: str
    wandb_proj_name: str
    wandb_run_name: str
    save_all_checkpoints: bool = False
    save_interval: int = 0  # if 0, only save best.model and checkpoint.last

    pretrained_model_path: Optional[str] = None

    # hyperparameters
    seed: int = 42
    num_train_epochs: int = 100
    early_stopping: int = 5  # if 0, don't stop training until maximum epoch
    data_size_percentage: int = 100
    per_gpu_train_batch_size: int = 16
    per_gpu_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 16
    adam_epsilon: float = 1e-8
    # weight_decay: float = 0.0
    use_grad_clip: bool = False
    max_grad_norm: float = 0.0
    warmup_steps: int = 5
    learning_rate: float = 2e-5
    # bert_dropout: float = 0.1  # dropout rate for Attention, FC in RoBERTa
    # embed_dropout: float = 0.0  # for last_hidden_state in SyntacticModel
    # layer_norm_eps: float = 1e-5

    # model specific parameters
    tokenize_model: str = "roberta-base"
    base_model: str = "roberta-base"
    model_type: str = "HolCCG"

    # additional parameters
    n_gpu: Optional[int] = None
    device: Optional[torch.device] = None
    pad_token_id: Optional[int] = None

    def set_additional_parameters(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        assert self.seed is not None
        assert self.n_gpu is not None

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)

        logger.info(f"set seed: {self.seed}")
        logger.info(f"device: {self.device}, n_gpu: {self.n_gpu}")

        self.learning_rate = float(self.learning_rate)

    #     self.adam_epsilon = float(self.adam_epsilon)
    #     if self.min_learning_rate is None:
    #         self.min_learning_rate = self.learning_rate / 20
    #     tokenizer = load_tokenizer(self.tokenize_type)
    #     self.pad_token_id = tokenizer.pad_token_id
    #     self.sep_token_id = tokenizer.sep_token_id


class TrainingComponents:
    def __init__(self):
        self.args: Optional[FinetuningArgs] = None
        # self.args: Optional[FinetuningArgs] = (None,)
        self.train_data_loader: Optional[torch.utils.data.DataLoader] = None
        self.dev_data_loader: Optional[torch.utils.data.DataLoader] = None

        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[Union[torch.optim.Adam, torch.optim.AdamW]] = None
        # self.lr_scheduler = None
        # self.training_scheduler: Optional[TrainingScheduler] = None

    def set_finetuning_components(self, args: FinetuningArgs):
        args.set_additional_parameters()
        self.args = args

        # load dataset
        assert Path(self.args.train_data_file).exists()
        self.train_data_loader = MyDataLoader(
            file_path=self.args.train_data_file,
            batch_size=self.args.per_gpu_train_batch_size,
            # n_max_tokens=self.args.per_gpu_train_max_tokens,
            data_size_percentage=self.args.data_size_percentage,
            shuffle=True,
            model_type=self.args.model_type,
        )

        assert Path(self.args.dev_data_file).exists()
        self.dev_data_loader = MyDataLoader(
            file_path=self.args.dev_data_file,
            batch_size=self.args.per_gpu_eval_batch_size,
            # n_max_tokens=self.args.per_gpu_train_max_tokens,
            shuffle=False,
            model_type=self.args.model_type,
        )

        # set model
        model: nn.Module
        config = RobertaConfig.from_pretrained(RoBERTaBase)
        base_model = RobertaModel.from_pretrained(RoBERTaBase)

        if self.args.model_type == BASECLS:
            model = BaseCLSModel(
                config=config,
                base_model=base_model,
                device=self.args.device,
            )
        elif self.args.model_type == BASEBIENCODER:
            base_model2 = RobertaModel.from_pretrained(RoBERTaBase)
            model = BaseBiEncoderModel(
                config=config,
                enc_model1=base_model,
                enc_model2=base_model2,
                device=self.args.device,
            )
        # elif self.args.model_type == HolCCG:
        #     model = HolCCGModel()
        else:
            raise ValueError(f"unsupported value: {args.model_type}")

        if self.args.pretrained_model_path:
            model.load_state_dict(torch.load(self.args.pretrained_model_path))

        if torch.cuda.is_available():
            model = model.cuda()
        self.model = model

        # set optimizer and scheduler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        # num_training_steps = (
        #     args.num_train_epochs
        #     * len(self.train_data_loader)
        #     // args.gradient_accumulation_steps
        # )
        # self.lr_scheduler = get_cosine_schedule_with_warmup(
        #     optimizer=self.optimizer,
        #     num_warmup_steps=args.warmup_steps,
        #     num_training_steps=num_training_steps,
        # )
        # if args.resume:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     scheduler.load_state_dict(checkpoint['scheduler'])
        # def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
        #     filepath = os.path.join(checkpoint, filename)
        #     torch.save(state, filepath)
        #     if is_best:
        #         shutil.copyfile(filepath, os.path.join(checkpoint,
        #                                                'model_best.pth.tar'))
        # save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': model_to_save.state_dict(),
        #         'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
        #         'acc': test_acc,
        #         'best_acc': best_acc,
        #         'optimizer': optimizer.state_dict(),
        #         'scheduler': scheduler.state_dict(),
        #     }, is_best, args.out)

        wandb_config = {
            "model_type": args.model_type,
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "train_batch_size": args.per_gpu_train_batch_size,
            "eval_batch_size": args.per_gpu_eval_batch_size,
        }
        wandb.init(
            project=args.wandb_proj_name, name=args.wandb_run_name, config=wandb_config
        )
        wandb.watch(models=self.model)


def finetuning(tc: TrainingComponents):
    args = tc.args
    total_batch_size = (
        args.per_gpu_train_batch_size * args.gradient_accumulation_steps * args.n_gpu
    )
    logger.info(f"Total batch size: {total_batch_size}")
    logger.info(f"Max epoch: {args.num_train_epochs}")

    for epoch in range(1, args.num_train_epochs + 1):
        logger.info(f"Epoch: {epoch}")

        # Training
        logger.info("Training")
        tc.model.train()
        logger.debug("calculating loss on train set")
        train_loss = compute_loss_finetuning(tc)
        logger.debug("evaluating on train set")
        train_eval_score = evaluate_finetuning(tc)
        logger.info(
            f"Train loss: {train_loss}, \
                    Train eval score: {train_eval_score}"
        )

        # Evaluation
        logger.info("Evaluation")
        tc.model.eval()
        logger.debug("calculating loss on dev set")
        with torch.no_grad():
            dev_loss = compute_loss_finetuning(tc)
        logger.debug("evaluating on dev set")
        dev_eval_score = evaluate_finetuning(tc)
        logger.info(
            f"Dev loss: {dev_loss}, \
                    Dev eval score: {dev_eval_score}"
        )
        wandb.log(
            {
                "epoch": epoch,
                # "lr": lr_groups,
                "train_loss": train_loss,
                "dev_loss": dev_loss,
                "train_score": train_eval_score,
                "dev_score": dev_eval_score,
            },
            commit=True,
        )


def compute_loss_finetuning(tc: TrainingComponents) -> float:
    args: FinetuningArgs = tc.args
    data_loader = tc.train_data_loader if tc.model.training else tc.dev_data_loader
    data_loader.create_batches()

    tc.optimizer.zero_grad()
    tc.model.zero_grad()

    total_loss = 0.0
    for n_iter, batch in enumerate(tqdm(data_loader), 1):
        loss = tc.model(batch)
        if tc.model.training:
            loss.backward()
            if n_iter % args.gradient_accumulation_steps == 0 or n_iter == len(
                data_loader
            ):
                if args.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        tc.model.parameters(), args.max_grad_norm  # type: ignore
                    )
                tc.optimizer.step()
                # tc.lr_scheduler.step()
                tc.model.zero_grad()
        else:
            loss = float(loss.cpu())
        total_loss += float(loss)

    assert n_iter == len(data_loader)
    ave_loss = total_loss / n_iter

    return ave_loss


def evaluate_finetuning(tc: TrainingComponents) -> float:
    data_loader = tc.train_data_loader if tc.model.training else tc.dev_data_loader
    data_loader.create_batches()

    total_inst_num = 0
    correct_entilment = 0
    correct_neutral = 0
    correct_contradiction = 0

    for batch in tqdm(data_loader):
        with torch.no_grad():
            prediction = tc.model.inference(batch)
            gold_label = batch["gold_label"]
            assert len(prediction) == len(gold_label)

            for pred, gold in zip(prediction, gold_label):
                if gold != 3:  # ignore '-'
                    total_inst_num += 1
                    if gold == 0 and pred == 0:
                        correct_entilment += 1
                    elif gold == 1 and pred == 1:
                        correct_neutral += 1
                    elif gold == 2 and pred == 2:
                        correct_contradiction += 1
                    else:
                        pass

    logger.info(f"total_inst_num: {total_inst_num}")
    logger.info("correct nums")
    logger.info(f"entilment: {correct_entilment}, \
                neutral: {correct_neutral}, \
                contradiction: {correct_contradiction}")

    micro_avg = (
        correct_entilment + correct_neutral + correct_contradiction
    ) / total_inst_num

    return micro_avg


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument(
        "--yaml_file", type=Path, required=True, help="Path to yaml file"
    )

    return parser


# @hydra.main(config_path="conf", config_name="config")
def main():
    parser = create_parser()
    args = parser.parse_args()

    params_dict = yaml.safe_load(open(args.yaml_file))

    # Setup logger
    output_dir = params_dict["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logzero.setup_logger(
        logfile=Path(output_dir) / Path(LOG_FILE_BASENAME),
        level=20,
    )

    assert "model_type" in params_dict, "error: 'model_type' doesn't exist"
    if params_dict["model_type"] in ModelTypes:
        params = FinetuningArgs(**params_dict)
        training_mode = "finetuning"
    else:
        raise ValueError(f'unsupported valuee: {params_dict["model_type"]}')

    # finetuning
    logger.info("training mode: %s", training_mode)
    if training_mode == "finetuning":
        training_components = TrainingComponents()
        training_components.set_finetuning_components(params)
        finetuning(training_components)

    logger.info("Training all done ...")


if __name__ == "__main__":
    main()
