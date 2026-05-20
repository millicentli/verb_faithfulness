"""
Trains inversion models in two settings:
(1) over all activations at a time (latentqa)
(2) over only the last-token activation (patchscopes)
"""

import os
from dataclasses import fields
import random
import fire
from tqdm import tqdm

import time

import numpy as np
import wandb
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig
import torch
import torch.optim as optim
import torch.distributed as dist

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True"
)

from src.configs.train_inversion_config import train_inversion_config
from src.configs.peft_config import lora_config
from src.utils.activation_utils import latent_qa
from src.utils.infra_utils import (
    get_logger,
    setup_wandb,
    get_ema,
    update_ema,
    update_config,
    get_modules,
)
from src.utils.inversion_utils import (
    get_model,
    get_tokenizer,
    save_model,
    get_dataloaders,
)

from src.utils.infra_utils import save_model as save_model_lqa
from src.utils.infra_utils import get_tokenizer as get_tokenizer_lqa
from src.utils.infra_utils import get_model as get_model_lqa

from src.utils.model_utils import MRec


def main(**kwargs):
    # Get args and setup DDP
    dist.init_process_group("nccl")
    assert torch.cuda.is_available()
    args = train_inversion_config()

    update_config(args, **kwargs)
    fsdp_args = None
    if args.use_fsdp:
        from src.configs.fsdp_config import fsdp_config

        fsdp_args = fsdp_config()
        update_config(fsdp_args, **kwargs)

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    seed = args.seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    logger = get_logger(args, rank)
    wandb_run = None
    if args.use_wandb and rank == 0:
        wandb_run = setup_wandb(args, fsdp_args, **kwargs)

    # Load tokenizer
    if args.method == "patchscopes":
        act_tokenizer = get_tokenizer(args.activation_model_name)
        reconstruct_tokenizer = get_tokenizer(args.reconstruct_model_name, reconstruct=True)
    elif args.method == "latentqa":
        act_tokenizer = get_tokenizer_lqa(args.activation_model_name)
        reconstruct_tokenizer = get_tokenizer_lqa(args.reconstruct_model_name)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # We use the same dataloading scheme
    train_dataloader, eval_dataloader = get_dataloaders(args, act_tokenizer, reconstruct_tokenizer)

    # Load the models
    lora_params = {
        k.name: getattr(lora_config(), k.name) for k in fields(lora_config())
    }
    peft_config = LoraConfig(**lora_params)
    if args.method == "patchscopes":
        act_model = get_model(
            args.activation_model_name,
            act_tokenizer,
            load_peft_checkpoint=None,
            fsdp_args=fsdp_args,
            device=device,
            rank=rank,
            extract_act=True,
            args=args,
        )
        reconstruct_model = get_model(
            args.reconstruct_model_name,
            reconstruct_tokenizer,
            peft_config=peft_config if args.load_model_checkpoint is None else None,
            load_peft_checkpoint=args.load_model_checkpoint,
            rank=rank,
            fsdp_args=fsdp_args,
            distributed_training=True,
            device="cuda",
            reconstruct=True,
            args=args,
        )
        reconstruct_model = MRec(reconstruct_model, reconstruct_tokenizer)
    else:
        act_model = get_model_lqa(
            args.activation_model_name, act_tokenizer, fsdp_args=fsdp_args, device=device, rank=rank
        )
        reconstruct_model = get_model_lqa(
            args.reconstruct_model_name,
            reconstruct_tokenizer,
            peft_config=peft_config,
            fsdp_args=fsdp_args,
            device=device,
            rank=rank,
            distributed_training=True,
        )
        module_read, module_write = get_modules(
            act_model, reconstruct_model, **args.__dict__
        )

    if hasattr(reconstruct_model, "model"):
        rec_module = reconstruct_model.model.module if hasattr(reconstruct_model.model, "module") else reconstruct_model.model
    elif hasattr(reconstruct_model, "module"):
        rec_module = reconstruct_model.module
    else:
        rec_module = reconstruct_model
    if rank == 0:
        if hasattr(rec_module, "print_trainable_parameters"):
            rec_module.print_trainable_parameters()
        if wandb_run is not None and args.load_model_checkpoint == "":
            wandb_run.config.update(peft_config)

    ema = get_ema(rec_module, decay=args.ema_decay, device=device)

    # Initialize the optimizer and learning rate scheduler
    optimizer = optim.AdamW(
        reconstruct_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    training_steps = len(train_dataloader) * args.num_epochs
    logger.info(f"Training steps: {training_steps}")
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=training_steps,
    )

    if args.load_model_checkpoint is not None:
        optim_ckpt = torch.load(os.path.join(args.load_model_checkpoint, "optimizer.pt"))
        scheduler_ckpt = torch.load(os.path.join(args.load_model_checkpoint, "scheduler.pt"))
        optimizer.load_state_dict(optim_ckpt)
        scheduler.load_state_dict(scheduler_ckpt)
        print("Loading optimizer and scheduler checkpoints at directory: ", args.load_model_checkpoint)
    else:
        print("No optimizer and scheduler checkpoints found, training from scratch!")

    # Start the training
    train_steps = 0
    train_start = time.time()
    stop_training = False
    for epoch in range(args.num_epochs):
        reconstruct_model.train()
        total_length = len(train_dataloader) // args.gradient_accumulation_steps
        pbar = tqdm(
            colour="blue",
            desc=f"Training Epoch: {epoch+1}",
            total=total_length,
            dynamic_ncols=True,
        )
        for step, batch in enumerate(train_dataloader):
            train_steps += 1
            if args.method == "patchscopes":
                with act_model.trace(batch["input_text"]) as tracer:
                    if hasattr(act_model, "transformer"):
                        clean_hs = act_model.transformer.h[args.layer_idx].output[0].save()
                    else:
                        clean_hs = act_model.model.layers[args.layer_idx].output[0].save()
                batch.pop("input_text")
                batch["activations"] = clean_hs[:, -1, :]
                for key in batch.keys():
                    batch[key] = batch[key].to(rank)
                outputs = reconstruct_model(**batch, use_cache=False)
            else:
                assert len(module_read) == len(module_write)
                assert len(module_read) == 1

                outputs = latent_qa(
                    batch,
                    act_model,
                    reconstruct_model,
                    module_read[0],
                    module_write[0],
                    act_tokenizer,
                    reconstruct_tokenizer,
                    truncate_verbs=None,
                    shift_position_ids=args.shift_position_ids,
                )

            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if train_steps % args.gradient_accumulation_steps == 0:
                if args.gradient_clipping and args.gradient_clipping_threshold > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        reconstruct_model.parameters(),
                        args.gradient_clipping_threshold,
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                update_ema(ema, rec_module, decay=args.ema_decay)
                pbar.update(1)

                if args.max_train_hours > 0:
                    elapsed_hours = (time.time() - train_start) / 3600
                    if elapsed_hours >= args.max_train_hours:
                        logger.info(
                            f"Reached time limit ({elapsed_hours:.2f}h >= {args.max_train_hours}h). Saving and stopping."
                        )
                        if args.method == "patchscopes":
                            save_model(
                                reconstruct_model,
                                ema,
                                reconstruct_tokenizer,
                                optimizer,
                                scheduler,
                                args,
                                epoch,
                                train_steps,
                                logger,
                                rank,
                            )
                        else:
                            save_model_lqa(
                                reconstruct_model if args.use_fsdp else reconstruct_model.module,
                                ema,
                                reconstruct_tokenizer,
                                args,
                                epoch,
                                train_steps,
                                logger,
                                rank,
                            )
                        stop_training = True
                        break

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/epoch": epoch,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss.detach().float(),
                    }
                )

            pbar.set_description(
                f"Training Epoch: {epoch+1}/{args.num_epochs}, batch {step+1}/{len(train_dataloader)} completed (loss: {loss.detach().float()})"
            )
            if args.eval_ppl and train_steps % args.eval_every_n_steps == 0:
                assert eval_dataloader is not None
                total_loss = 0.0
                pbar = tqdm(
                    colour="green",
                    desc=f"Evaluating Epoch: {epoch+1}",
                    total=len(eval_dataloader),
                    dynamic_ncols=True,
                )
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        if args.method == "patchscopes":
                            with act_model.trace(batch["input_text"]) as tracer:
                                if hasattr(act_model, "transformer"):
                                    clean_hs = act_model.transformer.h[args.layer_idx].output[0].save()
                                else:
                                    clean_hs = act_model.model.layers[args.layer_idx].output[0].save()
                            batch.pop("input_text")
                            batch["activations"] = clean_hs[:, -1, :]
                            for key in batch.keys():
                                batch[key] = batch[key].to(rank)
                            outputs = reconstruct_model(**batch, use_cache=False)
                        else:
                            assert len(module_read) == len(module_write)
                            assert len(module_read) == 1
                            outputs = latent_qa(
                                batch,
                                act_model,
                                reconstruct_model,
                                module_read[0],
                                module_write[0],
                                act_tokenizer,
                                reconstruct_tokenizer,
                                truncate_verbs=None,
                                shift_position_ids=args.shift_position_ids,
                            )

                        total_loss += outputs.loss.detach().float()
                        pbar.update(1)

                losses = torch.zeros(8).to(f"cuda:{rank}")
                losses[rank] = total_loss
                gathered_loss = (
                    [torch.empty_like(losses) for _ in range(dist.get_world_size())]
                    if rank == 0
                    else None
                )
                dist.gather(losses, gathered_loss, dst=0)
                if rank == 0 and wandb_run is not None:
                    all_loss = torch.sum(torch.stack(gathered_loss))
                    all_loss = all_loss / len(eval_dataloader) / dist.get_world_size()
                    wandb_run.log(
                        {
                            "train/epoch": epoch,
                            "train/step": epoch * len(train_dataloader) + step,
                            "eval/loss": all_loss.detach().float(),
                        }
                    )

            if train_steps % args.save_every_n_steps == 0:
                if args.method == "patchscopes":
                    save_model(
                        reconstruct_model,
                        ema,
                        reconstruct_tokenizer,
                        optimizer,
                        scheduler,
                        args,
                        epoch,
                        train_steps,
                        logger,
                        rank,
                    )
                else:
                    save_model_lqa(
                        reconstruct_model if args.use_fsdp else reconstruct_model.module,
                        ema,
                        reconstruct_tokenizer,
                        args,
                        epoch,
                        train_steps,
                        logger,
                        rank,
                    )

        pbar.close()
        if stop_training:
            break

        # End of epoch

        if args.save_model:
            if args.method == "patchscopes":
                save_model(
                    reconstruct_model,
                    ema,
                    reconstruct_tokenizer,
                    optimizer,
                    scheduler,
                    args,
                    epoch,
                    train_steps,
                    logger,
                    rank,
                )
            else:
                save_model_lqa(
                    reconstruct_model if args.use_fsdp else reconstruct_model.module,
                    ema,
                    reconstruct_tokenizer,
                    args,
                    epoch,
                    train_steps,
                    logger,
                    rank,
                )
            dist.barrier()

    if wandb_run is not None:
        wandb.finish()
    dist.destroy_process_group()
    logger.info("Training completed!")


if __name__ == "__main__":
    fire.Fire(main)
