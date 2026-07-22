import argparse
import json
import os
import warnings

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Harmless DDP perf note triggered by the depthwise conv inside SegFormer's
# MixFFN (the 2048-ch [2048,1,3,3] grad): the size-1 dim makes cuDNN report a
# non-contiguous grad stride that points at the same memory DDP's bucket expects.
# Not an error and does not affect correctness or mIoU; silenced to keep logs clean.
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")

from data import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    SUNDataset,
    apply_protocol,
    get_angle_grad_target,
    parse_scales,
    protocol_class_names,
)
from rsgnet import RSGNet, RSGNetLoss, edge_target_from_mask, reliability_calibration_loss
from utils import (
    build_ema_model,
    cleanup_distributed,
    compute_class_weights,
    freeze_encoder_bn,
    is_dist_initialized,
    is_main_process,
    rampup_factor,
    set_random_seed,
    setup_logger,
    validate,
)


CODE_VERSION = "RSGNet-20260714-physics-factorized-compatible"


class DistributedEvalSampler(Sampler):
    """Shard evaluation data across ranks without padding or duplication."""

    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.num_replicas - 1) // self.num_replicas)


def strip_module_prefix(state_dict):
    stripped = {}
    for key, value in state_dict.items():
        if key == "n_averaged":
            continue
        while key.startswith("module."):
            key = key[7:]
        stripped[key] = value
    return stripped


def extract_model_state(checkpoint, prefer_ema=False):
    if not isinstance(checkpoint, dict):
        return checkpoint
    if prefer_ema and checkpoint.get("ema_state_dict") is not None:
        return checkpoint["ema_state_dict"]
    for key in ("model_state_dict", "ema_state_dict", "state_dict"):
        if checkpoint.get(key) is not None:
            return checkpoint[key]
    return checkpoint


def filter_shape_mismatch(state_dict, model_state_dict):
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state_dict and model_state_dict[key].shape != value.shape:
            skipped.append((key, tuple(value.shape), tuple(model_state_dict[key].shape)))
            continue
        filtered[key] = value
    return filtered, skipped


def get_ema_state_dict(ema_model):
    if ema_model is None:
        return None
    ema_raw = ema_model.module if hasattr(ema_model, "module") else ema_model
    return ema_raw.state_dict()


def save_training_checkpoint(
    path,
    epoch,
    raw_model,
    ema_model,
    optimizer,
    scheduler,
    best_miou,
    config=None,
    include_training_state=True,
):
    payload = {
        "epoch": epoch,
        "model_state_dict": raw_model.state_dict(),
        "ema_state_dict": get_ema_state_dict(ema_model),
        "best_miou": best_miou,
        "code_version": CODE_VERSION,
        "config": dict(config) if config is not None else None,
    }
    if include_training_state:
        payload.update({
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        })
    torch.save(payload, path)


def save_best_checkpoints(
    save_dir,
    epoch,
    raw_model,
    ema_model,
    optimizer,
    scheduler,
    best_miou,
    config=None,
):
    miou_path = os.path.join(save_dir, f"best_miou_{best_miou:.4f}_epoch_{epoch:03d}.pth")
    save_training_checkpoint(
        miou_path,
        epoch,
        raw_model,
        ema_model,
        optimizer,
        scheduler,
        best_miou,
        config=config,
        include_training_state=False,
    )

    for filename in os.listdir(save_dir):
        if (
            (filename.startswith("best_miou_") and filename.endswith(".pth") and filename != os.path.basename(miou_path))
            or filename == "best_model.pth"
        ):
            try:
                os.remove(os.path.join(save_dir, filename))
            except OSError:
                pass
    return miou_path


def reduce_optimizer_lr(optimizer, scheduler, factor, min_lr):
    changed = False
    for group in optimizer.param_groups:
        old_lr = float(group["lr"])
        new_lr = max(float(min_lr), old_lr * float(factor))
        if new_lr < old_lr:
            group["lr"] = new_lr
            changed = True

    if hasattr(scheduler, "base_lrs"):
        scheduler.base_lrs = [max(float(min_lr), float(base_lr) * float(factor)) for base_lr in scheduler.base_lrs]

    return changed


def apply_pure_ce_overrides(args):
    if not args.pure_ce:
        return
    args.lambda_lovasz = 0.0
    args.lambda_dice = 0.0
    args.lambda_edge = 0.0
    args.lambda_boundary = 0.0
    args.lambda_feature_precision = 0.0
    args.hha_edge_weight = 0.0
    args.ohem_min_kept = max(args.ohem_min_kept, 2_000_000)
    args.ohem_start_epoch = max(args.ohem_start_epoch, args.epochs + 1)
    args.ce_only_epochs = max(args.ce_only_epochs, args.epochs)
    args.aux_weight = 0.0
    args.lambda_reliability = 0.0


def build_optimizer(model, lr, encoder_lr_mult=0.1):
    raw_model = model.module if hasattr(model, "module") else model
    assigned_param_ids = set()

    def collect_params(prefixes):
        params = []
        for name, param in raw_model.named_parameters():
            if not param.requires_grad or id(param) in assigned_param_ids:
                continue
            if any(name.startswith(prefix) for prefix in prefixes):
                params.append(param)
                assigned_param_ids.add(id(param))
        return params

    rgb_params = collect_params(("encoder.rgb_backbone.",))
    prompt_params = collect_params(
        (
            "encoder.reliability_head.",
            "encoder.structure_encoder.",
            "encoder.edge_descriptor.",
            "encoder.guide_pyramid.",
            "encoder.prompt_encoder.",
            "encoder.reliability_gate.",
        )
    )
    fusion_params = collect_params(
        (
            "encoder.fusion_stages.",
            "encoder.cross_scale_aligners.",
            "encoder.context_head.",
        )
    )
    head_params = [
        param for param in raw_model.parameters()
        if param.requires_grad and id(param) not in assigned_param_ids
    ]

    return torch.optim.AdamW(
        [
            {"params": rgb_params, "lr": lr * encoder_lr_mult},
            {"params": prompt_params, "lr": lr * 1.0},
            {"params": fusion_params, "lr": lr * 1.0},
            {"params": head_params, "lr": lr},
        ],
        weight_decay=1e-4,
    )


def build_scheduler(optimizer, epochs, train_loader_len, batch_size, world_size=1,
                    target_global_batch=16, warmup_epochs=5, min_lr_ratio=0.0):
    # Keep the effective (global) batch fixed at ``target_global_batch`` regardless
    # of how many GPUs are used, so single-GPU and multi-GPU runs share the same
    # optimization trajectory. accumulation = target / (per_gpu_batch * world_size).
    effective_batch = max(1, batch_size * max(1, world_size))
    accumulation_steps = max(1, target_global_batch // effective_batch)
    actual_steps_per_epoch = train_loader_len // accumulation_steps + (1 if train_loader_len % accumulation_steps != 0 else 0)
    total_steps = epochs * actual_steps_per_epoch
    warmup_steps = max(0, warmup_epochs) * actual_steps_per_epoch
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))

    def warmup_poly_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            warmup_start = 0.05
            return warmup_start + (1.0 - warmup_start) * float(step) / float(max(1, warmup_steps))
        decay_steps = max(1, total_steps - warmup_steps)
        poly = (1.0 - (step - warmup_steps) / decay_steps) ** 0.9
        return min_lr_ratio + (1.0 - min_lr_ratio) * poly

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_poly_lambda)
    return scheduler, accumulation_steps


def main(args):
    apply_pure_ce_overrides(args)
    if args.lambda_reliability > 0 and args.disable_reliability:
        raise ValueError(
            "--lambda-reliability > 0 requires reliability guidance; remove "
            "--disable-reliability or set --lambda-reliability 0."
        )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1 and not args.no_data_parallel and torch.cuda.is_available()

    if use_ddp:
        visible_cuda_devices = torch.cuda.device_count()
        if local_rank >= visible_cuda_devices:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {visible_cuda_devices} CUDA device(s) are visible. "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}."
            )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
        print(
            f"[rank {dist.get_rank()}] device={device} "
            f"name={torch.cuda.get_device_name(local_rank)} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
            flush=True,
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger(args.save_dir, enabled=is_main_process())
    if is_main_process():
        logger.info("=" * 50)
        logger.info(f"Code version: {CODE_VERSION}")
        logger.info(f"Training config: {args}")
        logger.info("=" * 50)

    set_random_seed(args.seed + (dist.get_rank() if is_dist_initialized() else 0))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = RSGNet(
        n_classes=args.n_classes,
        pretrained_path=args.pretrained_encoder,
        return_aux=True,
        encoder_name=args.encoder_name,
        drop_path_rate=args.drop_path_rate,
        prompt_channels=args.prompt_channels,
        decoder_channels=args.decoder_channels,
        attention_tokens=args.attention_tokens,
        local_scale_init=args.local_scale_init,
        semantic_scale_init=args.semantic_scale_init,
        fusion_mode=args.fusion_mode,
        use_reliability=not args.disable_reliability,
        architecture_variant=args.architecture_variant,
        geometry_encoding=args.geometry_encoding,
        geometry_channels=args.geometry_channels,
    ).to(device)
    if is_main_process():
        logger.info(
            "RSGNet architecture | "
            f"variant={args.architecture_variant} "
            f"geometry_encoding={args.geometry_encoding} "
            f"fusion_mode={args.fusion_mode} "
            f"reliability={not args.disable_reliability} "
            f"prompt_channels={args.prompt_channels} "
            f"decoder_channels={args.decoder_channels} "
            f"attention_tokens={args.attention_tokens}"
        )
        logger.info(
            "Residual initialization | "
            f"local={args.local_scale_init:g} semantic={args.semantic_scale_init:g}"
        )

    checkpoint = None
    start_epoch = 0
    best_miou = args.initial_best_miou
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location="cpu")
        checkpoint_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
        checkpoint_val_fraction = (
            checkpoint_config.get("val_fraction")
            if isinstance(checkpoint_config, dict)
            else None
        )
        checkpoint_variant = (
            checkpoint_config.get("architecture_variant", "legacy")
            if isinstance(checkpoint_config, dict)
            else "legacy"
        )
        if checkpoint_variant != args.architecture_variant:
            raise ValueError(
                "Checkpoint architecture mismatch: "
                f"checkpoint={checkpoint_variant!r}, requested={args.architecture_variant!r}. "
                "A, B, and C architecture checkpoints are not interchangeable."
            )
        if args.architecture_variant == "factorized" and isinstance(checkpoint_config, dict):
            checkpoint_encoding = checkpoint_config.get(
                "geometry_encoding", "factorized_routed"
            )
            compatible_screening_encodings = {
                "factorized_routed",
                "factorized_final",
                "factorized_final_soft",
                "factorized_channel_routed",
            }
            if (
                checkpoint_encoding != args.geometry_encoding
                and not {
                    checkpoint_encoding,
                    args.geometry_encoding,
                }.issubset(compatible_screening_encodings)
            ):
                raise ValueError(
                    "Checkpoint geometry encoding mismatch: "
                    f"checkpoint={checkpoint_encoding!r}, requested={args.geometry_encoding!r}."
                )
            checkpoint_source = checkpoint_config.get("geometry_source", "hha")
            if checkpoint_source != args.geometry_source:
                raise ValueError(
                    "Checkpoint geometry source mismatch: "
                    f"checkpoint={checkpoint_source!r}, requested={args.geometry_source!r}."
                )
            checkpoint_channels = checkpoint_config.get("geometry_channels", "dha")
            if checkpoint_channels != args.geometry_channels:
                raise ValueError(
                    "Checkpoint geometry channel mask mismatch: "
                    f"checkpoint={checkpoint_channels!r}, requested={args.geometry_channels!r}."
                )
        checkpoint_has_training_state = (
            isinstance(checkpoint, dict)
            and "optimizer_state_dict" in checkpoint
            and "scheduler_state_dict" in checkpoint
        )
        prefer_ema_weights = args.resume_weights_only or not checkpoint_has_training_state
        state_dict = strip_module_prefix(
            extract_model_state(checkpoint, prefer_ema=prefer_ema_weights)
        )
        state_dict, skipped_mismatch = filter_shape_mismatch(state_dict, model.state_dict())
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if is_main_process():
            logger.info(f"Loaded checkpoint weights from: {args.resume}")
            if (
                args.resume_weights_only
                and args.val_fraction > 0
                and (checkpoint_val_fraction is None or float(checkpoint_val_fraction) <= 0)
            ):
                logger.warning(
                    "VALIDATION OVERLAP: the warm-start checkpoint was trained without a "
                    "held-out validation split (or its split is unknown), while this run "
                    "holds out part of the same official training set. The checkpoint may "
                    "already have seen those validation images; resulting validation mIoU "
                    "is not valid for model selection or publication."
                )
            if skipped_mismatch:
                logger.warning(f"Skipped shape-mismatch keys while loading checkpoint: {len(skipped_mismatch)}")
                for key, old_shape, new_shape in skipped_mismatch[:20]:
                    logger.warning(f"  {key}: checkpoint={old_shape} model={new_shape}")
                if len(skipped_mismatch) > 20:
                    logger.warning(f"  ... and {len(skipped_mismatch) - 20} more")
            if missing:
                logger.warning(f"Missing keys while loading checkpoint: {len(missing)}")
            if unexpected:
                logger.warning(f"Unexpected keys while loading checkpoint: {len(unexpected)}")

    if use_ddp:
        # The network is BatchNorm-heavy; with a small per-GPU batch each rank
        # would otherwise normalize on its own ~2 samples, giving noisy stats and
        # a markedly lower multi-GPU mIoU. SyncBatchNorm aggregates statistics
        # across all ranks so the effective BN batch equals the global batch.
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=args.ddp_find_unused_parameters,
        )
        if is_main_process():
            logger.info(
                "DistributedDataParallel + SyncBatchNorm enabled "
                f"with world_size={world_size} "
                f"find_unused_parameters={args.ddp_find_unused_parameters}"
            )

    # Keep EMA on every rank so validation can be sharded across all GPUs.
    # DDP keeps the raw weights identical, therefore each EMA copy evolves
    # identically without additional synchronization.
    ema_model = build_ema_model(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_dataset = SUNDataset(
        args.data_root,
        dataset_name=args.dataset_name,
        mode="train",
        crop_size=(args.input_height, args.input_width),
        n_classes=args.n_classes,
        label_map_path=args.label_map,
        label_dir_name=args.label_dir_name,
        hha_dir_name=args.hha_dir_name,
        hha_name_prefix=args.hha_name_prefix,
        hha_channel_order=args.hha_channel_order,
        geometry_source=args.geometry_source,
        depth_dir_name=args.depth_dir_name,
        depth_name_prefix=args.depth_name_prefix,
        depth_scale=args.depth_scale,
        max_depth=args.max_depth,
        label_name_prefix=args.label_name_prefix,
        ignore_index=args.ignore_index,
        train_cutout_prob=args.train_cutout_prob,
        aug_level=args.aug_level,
        split_policy=args.split_policy,
        hha_degrade_prob=args.hha_degrade_prob,
        hha_degrade_modes=tuple(item.strip() for item in args.hha_degrade_modes.split(",") if item.strip()),
        reliability_clean_weight=args.reliability_clean_weight,
        reliability_target_temperature=args.reliability_target_temperature,
        return_reliability_target=args.lambda_reliability > 0,
        factorized_reliability_target=args.architecture_variant == "factorized",
        validation_fraction=args.val_fraction,
        validation_seed=args.val_seed,
        rare_crop_probability=args.rare_crop_probability,
        rare_crop_class_ids=args.rare_crop_class_ids,
        rare_crop_trials=args.rare_crop_trials,
    )
    val_dataset = SUNDataset(
        args.data_root,
        dataset_name=args.dataset_name,
        mode="val" if args.val_fraction > 0 else "test",
        crop_size=(args.input_height, args.input_width),
        n_classes=args.n_classes,
        label_map_path=args.label_map,
        label_dir_name=args.label_dir_name,
        hha_dir_name=args.hha_dir_name,
        hha_name_prefix=args.hha_name_prefix,
        hha_channel_order=args.hha_channel_order,
        geometry_source=args.geometry_source,
        depth_dir_name=args.depth_dir_name,
        depth_name_prefix=args.depth_name_prefix,
        depth_scale=args.depth_scale,
        max_depth=args.max_depth,
        label_name_prefix=args.label_name_prefix,
        ignore_index=args.ignore_index,
        split_policy=args.split_policy,
        eval_native_size=args.eval_native_size,
        eval_pad_size=(args.eval_pad_height, args.eval_pad_width)
        if args.eval_pad_height > 0 and args.eval_pad_width > 0
        else None,
        validation_fraction=args.val_fraction,
        validation_seed=args.val_seed,
    )
    if is_main_process():
        validation_source = "held-out train split" if args.val_fraction > 0 else "official test split"
        logger.info(
            f"Dataset partition | train={len(train_dataset)} val={len(val_dataset)} "
            f"source={validation_source} val_fraction={args.val_fraction:.4f} "
            f"val_seed={args.val_seed}"
        )

    train_sampler = None
    val_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=True,
        )
        val_sampler = DistributedEvalSampler(
            val_dataset,
            num_replicas=world_size,
            rank=dist.get_rank(),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1 if args.eval_native_size else args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        **loader_kwargs,
    )

    class_weights = None
    if args.use_class_weights:
        cache_path = os.path.join(args.save_dir, f"class_weights_{args.dataset_name}_{args.label_dir_name}_{args.n_classes}.json")
        if is_main_process():
            class_weights = compute_class_weights(
                train_dataset,
                n_classes=args.n_classes,
                ignore_index=args.ignore_index,
                miou_start_class=args.miou_start_class,
                mode=args.class_weight_mode,
                clamp_max=args.class_weight_clamp,
                cache_path=cache_path,
            )
        if use_ddp:
            if class_weights is None:
                class_weights = torch.zeros(args.n_classes, dtype=torch.float32)
            class_weights = class_weights.to(device)
            dist.broadcast(class_weights, src=0)
            class_weights = class_weights.cpu()
        if is_main_process():
            wt = class_weights
            nz = wt[wt > 0]
            logger.info(
                f"Class weights ({args.class_weight_mode}) | nonzero={int((wt > 0).sum())} "
                f"min={nz.min().item():.3f} max={wt.max().item():.3f} mean={nz.mean().item():.3f}"
            )

    criterion = RSGNetLoss(
        lambda_lovasz=args.lambda_lovasz,
        lambda_dice=args.lambda_dice,
        lambda_edge=args.lambda_edge,
        lambda_boundary=args.lambda_boundary,
        lambda_feature_precision=args.lambda_feature_precision,
        hha_edge_weight=args.hha_edge_weight,
        max_edge_pos_weight=args.max_edge_pos_weight,
        ohem_min_kept=args.ohem_min_kept,
        boundary_ce_weight=args.boundary_ce_weight,
        aux_weight=args.aux_weight,
        feature_precision_max_size=args.feature_precision_max_size,
        n_classes=args.n_classes,
        ignore_index=args.ignore_index,
        class_weights=class_weights,
    ).to(device)

    optimizer = build_optimizer(model, args.lr, encoder_lr_mult=args.encoder_lr_mult)
    if is_main_process():
        logger.info(
            f"Optimizer LR | base={args.lr:.2e} encoder_mult={args.encoder_lr_mult:.3f} "
            f"encoder={args.lr * args.encoder_lr_mult:.2e}"
        )
    scheduler, accumulation_steps = build_scheduler(
        optimizer,
        args.epochs,
        len(train_loader),
        args.batch_size,
        world_size=world_size if use_ddp else 1,
        target_global_batch=args.target_global_batch,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
    )

    if checkpoint is not None and "model_state_dict" in checkpoint:
        has_training_state = (
            "optimizer_state_dict" in checkpoint and "scheduler_state_dict" in checkpoint
        )
        restore_training_state = (not args.resume_weights_only) and has_training_state
        start_epoch = checkpoint.get("epoch", 0) if restore_training_state else 0
        best_miou = (
            args.initial_best_miou
            if args.resume_weights_only
            else max(checkpoint.get("best_miou", 0.0), args.initial_best_miou)
        )
        if restore_training_state:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if ema_model is not None and checkpoint.get("ema_state_dict") is not None:
            ema_state = strip_module_prefix(checkpoint["ema_state_dict"])
            ema_target = ema_model.module if hasattr(ema_model, "module") else ema_model
            ema_state, skipped_ema_mismatch = filter_shape_mismatch(ema_state, ema_target.state_dict())
            ema_target.load_state_dict(ema_state, strict=False)
            if is_main_process() and skipped_ema_mismatch:
                logger.warning(f"Skipped EMA shape-mismatch keys while loading checkpoint: {len(skipped_ema_mismatch)}")
        if is_main_process():
            if args.resume_weights_only or not has_training_state:
                logger.info(
                    f"Fine-tuning from checkpoint weights only; optimizer/scheduler reset, "
                    f"best mIoU floor={best_miou:.4f}"
                )
            else:
                logger.info(f"Resumed training from epoch {start_epoch}, best mIoU={best_miou:.4f}")
    elif checkpoint is not None and is_main_process():
        logger.warning(
            "Resume checkpoint contains weights only; optimizer/scheduler/epoch/best_miou were not restored."
        )

    epochs_without_improvement = 0
    epochs_without_best = 0
    auto_lr_reductions = 0

    amp_enabled = (not args.no_amp) and device.type == "cuda"
    if amp_enabled and args.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        if is_main_process():
            logger.warning("bf16 requested but unsupported on this device; falling back to fp16.")
        args.amp_dtype = "fp16"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    if is_main_process():
        logger.info(f"AMP: enabled={amp_enabled}, dtype={args.amp_dtype}, grad_scaler={scaler.is_enabled()}")

    try:
        for epoch in range(start_epoch, args.epochs):
            stop_training = False
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            ce_only = epoch < args.ce_only_epochs
            use_ohem = epoch >= args.ohem_start_epoch
            curriculum_epoch = max(0, epoch - args.ce_only_epochs)
            edge_loss_scale = 0.0 if ce_only else rampup_factor(curriculum_epoch, args.edge_warmup_epochs, args.loss_warmup_start)
            boundary_loss_scale = 0.0 if ce_only else rampup_factor(curriculum_epoch, args.boundary_warmup_epochs, args.loss_warmup_start)
            feature_loss_scale = 0.0 if ce_only else rampup_factor(curriculum_epoch, args.feature_warmup_epochs, args.loss_warmup_start)
            reliability_loss_scale = rampup_factor(epoch, args.reliability_warmup_epochs, args.loss_warmup_start)

            if is_main_process():
                logger.info(
                    f"Loss curriculum | ce_only={ce_only} ohem={use_ohem} "
                    f"edge={edge_loss_scale:.3f} boundary={boundary_loss_scale:.3f} "
                    f"feature={feature_loss_scale:.3f} reliability={reliability_loss_scale:.3f}"
                )

            model.train()
            freeze_encoder_bn(model)
            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{args.epochs}",
                disable=use_ddp and not is_main_process(),
                dynamic_ncols=True,
            )
            optimizer.zero_grad(set_to_none=True)

            epoch_loss_sum = 0.0
            epoch_loss_count = 0.0
            skipped_nonfinite = 0

            for step, batch in enumerate(progress):
                if len(batch) == 4:
                    rgb, hha, masks, reliability_supervision = batch
                else:
                    rgb, hha, masks = batch
                    reliability_supervision = None
                rgb = rgb.to(device, non_blocking=True)
                hha = hha.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                if reliability_supervision is not None:
                    reliability_supervision = reliability_supervision.to(device, non_blocking=True)
                masks[(masks != args.ignore_index) & ((masks < 0) | (masks >= args.n_classes))] = args.ignore_index

                valid_edge_mask = (masks != args.ignore_index).float().unsqueeze(1)
                edge_tgt = edge_target_from_mask(masks, ignore_index=args.ignore_index).float()
                angle_grad_tgt = get_angle_grad_target(hha).float() * valid_edge_mask

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    if reliability_supervision is not None and args.lambda_reliability > 0:
                        seg_logits, edge_logits, aux_logits, encoder_context = model(
                            rgb,
                            hha,
                            return_context=True,
                            hha_invalid_hint=reliability_supervision[:, -1:],
                        )
                    else:
                        seg_logits, edge_logits, aux_logits = model(rgb, hha)
                        encoder_context = None
                    raw_outputs = [seg_logits, edge_logits]
                    if aux_logits is not None:
                        raw_outputs.append(aux_logits)
                    has_nonfinite_outputs = not all(torch.isfinite(t).all() for t in raw_outputs)
                    nonfinite_flag = torch.tensor(
                        1 if has_nonfinite_outputs else 0,
                        device=device,
                        dtype=torch.int32,
                    )
                    if use_ddp:
                        dist.all_reduce(nonfinite_flag, op=dist.ReduceOp.MAX)
                    if int(nonfinite_flag.item()) > 0:
                        skipped_nonfinite += 1
                        optimizer.zero_grad(set_to_none=True)
                        del raw_outputs, seg_logits, edge_logits, aux_logits, nonfinite_flag
                        continue
                    total_loss, loss_ce, loss_lov, loss_dice, loss_edge, loss_feature_precision = criterion(
                        seg_logits,
                        masks,
                        edge_logits,
                        edge_tgt,
                        aux_logits=aux_logits,
                        hha_grad_target=angle_grad_tgt,
                        valid_edge_mask=valid_edge_mask,
                        edge_loss_scale=edge_loss_scale,
                        boundary_loss_scale=boundary_loss_scale,
                        feature_loss_scale=feature_loss_scale,
                        ce_only=ce_only,
                        use_ohem=use_ohem,
                    )
                    loss_reliability = loss_ce.detach() * 0.0
                    if encoder_context is not None:
                        reliability_target = reliability_supervision[:, :-2]
                        reliability_weight = reliability_supervision[:, -2:-1] * valid_edge_mask
                        loss_reliability = reliability_calibration_loss(
                            encoder_context["geometry_reliability"],
                            reliability_target,
                            reliability_weight,
                        )
                        total_loss = total_loss + (
                            args.lambda_reliability * reliability_loss_scale * loss_reliability
                        )

                if not torch.isfinite(total_loss):
                    skipped_nonfinite += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue

                is_accum_boundary = (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader)
                scaled_loss = scaler.scale(total_loss / accumulation_steps)
                # During gradient accumulation, only the boundary micro-step needs
                # to all-reduce gradients across ranks. Skipping the sync on the
                # intermediate steps removes (accumulation_steps - 1) redundant
                # all-reduces per optimizer update, the main DDP slowdown here.
                if use_ddp and not is_accum_boundary:
                    with model.no_sync():
                        scaled_loss.backward()
                else:
                    scaled_loss.backward()

                if is_accum_boundary:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if (not scaler.is_enabled()) or scaler.get_scale() >= scale_before:
                        scheduler.step()
                        if ema_model is not None:
                            raw_model = model.module if hasattr(model, "module") else model
                            ema_model.update_parameters(raw_model)
                    optimizer.zero_grad(set_to_none=True)

                epoch_loss_sum += total_loss.item()
                epoch_loss_count += 1.0
                if not (use_ddp and not is_main_process()):
                    progress.set_postfix(
                        tot=f"{total_loss.item():.3f}",
                        ce=f"{loss_ce.item():.3f}",
                        lov=f"{loss_lov.item():.3f}",
                        edg=f"{loss_edge.item():.3f}",
                        rel=f"{loss_reliability.item():.3f}",
                    )

            epoch_stats = torch.tensor([epoch_loss_sum, epoch_loss_count, float(skipped_nonfinite)], device=device)
            if use_ddp:
                dist.all_reduce(epoch_stats, op=dist.ReduceOp.SUM)
            avg_loss = (epoch_stats[0] / epoch_stats[1].clamp_min(1.0)).item()

            if is_main_process():
                skip_msg = f" | skipped_nonfinite={int(epoch_stats[2].item())}" if epoch_stats[2].item() > 0 else ""
                logger.info(
                    f"Epoch {epoch + 1}/{args.epochs} | Avg Train Loss: {avg_loss:.4f} | "
                    f"LR: {optimizer.param_groups[-1]['lr']:.8f}{skip_msg}"
                )

            raw_model = model.module if hasattr(model, "module") else model
            should_validate = (epoch + 1) % args.val_interval == 0 or (epoch + 1) == args.epochs
            if should_validate:
                if use_ddp:
                    dist.barrier(device_ids=[local_rank])
                lr_reduce_now = False
                use_ema_eval = ema_model is not None and (epoch + 1) >= args.ema_warmup_epochs
                eval_model = ema_model if use_ema_eval else raw_model
                val_scales = parse_scales(args.tta_scales) if args.use_tta else (1.0,)
                metrics = validate(
                    eval_model,
                    val_loader,
                    device=device,
                    n_classes=args.n_classes,
                    scales=val_scales,
                    use_flip=args.use_tta,
                    ignore_index=args.ignore_index,
                    miou_start_class=args.miou_start_class,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    crop_size=(args.eval_crop_height, args.eval_crop_width) if args.sliding_eval else None,
                    stride_rate=args.eval_stride_rate,
                    tta_fusion=args.tta_fusion,
                    tta_align_corners=args.tta_align_corners,
                    tta_size_mode=args.tta_size_mode,
                    tta_small_image_mode=args.tta_small_image_mode,
                    return_details=True,
                )
                miou = metrics["miou"]
                if is_main_process():
                    eval_name = "EMA" if use_ema_eval else "raw"
                    logger.info(
                        f"Epoch {epoch + 1} | Validation ({eval_name}) | "
                        f"mIoU={miou:.4f} interior={metrics['interior_miou']:.4f} "
                        f"boundary_f1={metrics['boundary_f1']:.4f}"
                    )

                    improved = miou > best_miou
                    if improved:
                        best_miou = miou
                        best_ckpt = save_best_checkpoints(
                            args.save_dir,
                            epoch + 1,
                            raw_model,
                            ema_model,
                            optimizer,
                            scheduler,
                            best_miou,
                            config=vars(args),
                        )
                        epochs_without_improvement = 0
                        epochs_without_best = 0
                        logger.info(
                            f"New best checkpoint reached: {best_miou:.4f} -> {best_ckpt}"
                        )
                        metrics.update({
                            "epoch": epoch + 1,
                            "protocol": args.protocol,
                            "architecture_variant": args.architecture_variant,
                            "geometry_encoding": args.geometry_encoding,
                            "hha_channel_order": args.hha_channel_order,
                            "geometry_source": args.geometry_source,
                            "geometry_channels": args.geometry_channels,
                            "fusion_mode": args.fusion_mode,
                            "reliability_enabled": not args.disable_reliability,
                            "scales": list(val_scales),
                            "flip": bool(args.use_tta),
                            "tta_fusion": args.tta_fusion,
                            "tta_align_corners": bool(args.tta_align_corners),
                            "tta_size_mode": args.tta_size_mode,
                            "tta_small_image_mode": args.tta_small_image_mode,
                            "eval_pad_size": [args.eval_pad_height, args.eval_pad_width],
                            "class_names": list(protocol_class_names(args.protocol, args.n_classes)),
                        })
                        metrics_path = os.path.join(args.save_dir, "best_metrics.json")
                        with open(metrics_path, "w", encoding="utf-8") as file:
                            json.dump(metrics, file, indent=2)
                        logger.info(f"Per-class metrics saved: {metrics_path}")
                    else:
                        epochs_without_improvement += 1
                        epochs_without_best += 1

                    if (
                        args.auto_lr
                        and (not improved)
                        and (epoch + 1) >= args.auto_lr_start_epoch
                        and epochs_without_improvement >= args.auto_lr_patience
                        and auto_lr_reductions < args.auto_lr_max_reductions
                    ):
                        lr_reduce_now = True
                if use_ddp:
                    lr_reduce_signal = torch.tensor([1 if lr_reduce_now else 0], device=device, dtype=torch.int64)
                    dist.broadcast(lr_reduce_signal, src=0)
                    lr_reduce_now = bool(lr_reduce_signal.item())
                if lr_reduce_now:
                    changed = reduce_optimizer_lr(
                        optimizer,
                        scheduler,
                        factor=args.auto_lr_factor,
                        min_lr=args.auto_lr_min,
                    )
                    if is_main_process() and changed:
                        auto_lr_reductions += 1
                        epochs_without_improvement = 0
                        logger.info(
                            "Auto LR refinement triggered | "
                            f"reduction={auto_lr_reductions}/{args.auto_lr_max_reductions} "
                            f"factor={args.auto_lr_factor} "
                            f"current_head_lr={optimizer.param_groups[-1]['lr']:.8f}"
                        )
                if is_main_process():
                    stop_training = (
                        args.early_stopping_patience > 0
                        and (epoch + 1) >= args.early_stopping_min_epoch
                        and epochs_without_best >= args.early_stopping_patience
                    )
                if use_ddp:
                    stop_signal = torch.tensor(
                        [1 if stop_training else 0], device=device, dtype=torch.int64
                    )
                    dist.broadcast(stop_signal, src=0)
                    stop_training = bool(stop_signal.item())
                if use_ddp:
                    dist.barrier(device_ids=[local_rank])

            if is_main_process():
                save_training_checkpoint(
                    os.path.join(args.save_dir, "latest_model.pth"),
                    epoch + 1,
                    raw_model,
                    ema_model,
                    optimizer,
                    scheduler,
                    best_miou,
                    config=vars(args),
                )
            if stop_training:
                if is_main_process():
                    logger.info(
                        "Early stopping triggered | "
                        f"epoch={epoch + 1} best_miou={best_miou:.4f} "
                        f"validations_without_best={epochs_without_best}"
                    )
                break
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/home/pengfei/HTCnet/DataSets")
    parser.add_argument("--dataset-name", type=str, default="SUNRGBD", help="Dataset folder under --data-root, e.g. SUNRGBD or NYU.")
    parser.add_argument(
        "--protocol",
        type=str,
        default="sunrgbd37",
        choices=("legacy", "sunrgbd37", "sunrgbd37_dformerv2"),
    )
    parser.add_argument("--save-dir", type=str, default="../Checkpoint_SUN_V8")
    parser.add_argument("--pretrained-encoder", type=str, default="/home/pengfei/HTCnet/Checkpoint/mit_b2.pth")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--resume-weights-only", action="store_true", help="Load checkpoint weights and best mIoU, but reset optimizer, scheduler, and epoch for fine-tuning.")
    parser.add_argument("--label-map", type=str, default="", help='Optional raw-label to train-label map: txt/csv lines "raw train" or JSON dict.')
    parser.add_argument("--split-policy", type=str, default="file", choices=("file", "sunrgbd_official"))
    parser.add_argument("--label-dir-name", type=str, default="Labels", help="Label folder under --data-root/--dataset-name, e.g. Labels, Labels40, or Label.")
    parser.add_argument("--hha-dir-name", type=str, default="HHA")
    parser.add_argument("--hha-name-prefix", type=str, default="", help="Optional filename prefix for HHA images, e.g. 'hha_' for NYU.")
    parser.add_argument(
        "--hha-channel-order",
        type=str,
        default="dha",
        choices=("dha", "ahd"),
        help="Channel order after RGB decoding; tensors are canonicalized to disparity-height-angle.",
    )
    parser.add_argument(
        "--geometry-source",
        type=str,
        default="hha",
        choices=("hha", "depth"),
        help="Use encoded HHA or a capacity-matched repeated metric-depth control.",
    )
    parser.add_argument("--depth-dir-name", type=str, default="Depth")
    parser.add_argument("--depth-name-prefix", type=str, default="")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--label-name-prefix", type=str, default="", help="Optional filename prefix for label images.")
    parser.add_argument("--train-cutout-prob", type=float, default=0.20,
                        help="Probability of RGB-only cutout during training. Lower it for late fine-tuning.")
    parser.add_argument("--aug-level", type=str, default="strong", choices=("base", "lsj", "strong"),
                        help="'base' keeps the original augmentation; 'strong' adds hue/brightness/contrast jitter, "
                             "wider scale range, and multi-block RGB+HHA cutout with label voiding.")
    parser.add_argument("--hha-degrade-prob", type=float, default=0.0,
                        help="Probability of synthetic HHA-only dropout, noise, or shift.")
    parser.add_argument("--hha-degrade-modes", type=str, default="dropout,noise,shift")
    parser.add_argument("--reliability-clean-weight", type=float, default=0.05,
                        help="Weak calibration weight for pixels not changed by synthetic degradation.")
    parser.add_argument("--reliability-target-temperature", type=float, default=0.08,
                        help="Temperature for synthetic reliability targets; larger values are less punitive.")
    parser.add_argument("--lambda-reliability", type=float, default=0.0,
                        help="Weight of dense reliability calibration loss; 0 keeps legacy training.")
    parser.add_argument("--reliability-warmup-epochs", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.0,
                        help="Deterministically hold out this fraction of the official training split for model selection.")
    parser.add_argument("--val-seed", type=int, default=3407)
    parser.add_argument("--rare-crop-probability", type=float, default=0.0,
                        help="Probability of selecting the best rare-class crop from multiple candidates.")
    parser.add_argument("--rare-crop-class-ids", type=str, default="",
                        help="Comma-separated train IDs targeted by class-aware crop selection.")
    parser.add_argument("--rare-crop-trials", type=int, default=8)
    parser.add_argument("--n-classes", type=int, default=41)
    parser.add_argument("--ignore-index", type=int, default=0, help="Void label id ignored by losses and metrics.")
    parser.add_argument("--miou-start-class", type=int, default=1, help="First class id included in mIoU.")
    parser.add_argument("--crop-size", type=int, default=None, help="Deprecated square input size. Prefer --input-height/--input-width.")
    parser.add_argument("--input-height", type=int, default=INPUT_HEIGHT)
    parser.add_argument("--input-width", type=int, default=INPUT_WIDTH)
    parser.add_argument("--allow-custom-input-size", action="store_true",
                        help="Keep explicit input dimensions instead of applying the protocol crop default.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-global-batch", type=int, default=16,
                        help="Effective batch after gradient accumulation, held fixed across GPU counts "
                             "so single-GPU and multi-GPU runs share the same optimization schedule.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--encoder-lr-mult", type=float, default=0.1,
                        help="LR multiplier for the pretrained RGB encoder parameter group.")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05, help="Keep LR above lr * min_lr_ratio near the end; useful for late fine-tuning.")
    parser.add_argument("--auto-lr", action="store_true", help="Automatically lower LR when validation mIoU plateaus during training.")
    parser.add_argument("--auto-lr-start-epoch", type=int, default=75, help="Do not trigger automatic LR reduction before this epoch.")
    parser.add_argument("--auto-lr-patience", type=int, default=10, help="Number of validations without improvement before lowering LR.")
    parser.add_argument("--auto-lr-factor", type=float, default=0.7, help="Multiplicative LR drop factor for automatic late refinement.")
    parser.add_argument("--auto-lr-min", type=float, default=1e-6, help="Absolute lower bound for automatically reduced LR.")
    parser.add_argument("--auto-lr-max-reductions", type=int, default=3, help="Maximum number of automatic LR reductions.")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--ema-decay", type=float, default=0.9996)
    parser.add_argument("--ema-warmup-epochs", type=int, default=10, help="Validate raw model before this epoch, then switch to EMA if enabled.")
    parser.add_argument("--no-amp", action="store_true", help="Disable native torch.autocast mixed precision training.")
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=("fp16", "bf16"))
    parser.add_argument("--no-data-parallel", action="store_true")
    parser.add_argument("--ddp-find-unused-parameters", action="store_true",
                        help="Enable DDP unused-parameter detection for ablations that bypass whole model branches.")
    parser.add_argument("--use-tta", action="store_true", help="Use multi-scale + flip augmentation during validation.")
    parser.add_argument("--tta-scales", type=str, default="0.75,1.0,1.25")
    parser.add_argument("--eval-native-size", action="store_true")
    parser.add_argument("--sliding-eval", action="store_true")
    parser.add_argument("--eval-crop-height", type=int, default=480)
    parser.add_argument("--eval-crop-width", type=int, default=480)
    parser.add_argument("--eval-stride-rate", type=float, default=2.0 / 3.0)
    parser.add_argument("--eval-pad-height", type=int, default=0)
    parser.add_argument("--eval-pad-width", type=int, default=0)
    parser.add_argument("--tta-fusion", choices=("logits", "probabilities"), default="logits")
    parser.add_argument("--tta-align-corners", action="store_true")
    parser.add_argument("--tta-size-mode", choices=("legacy", "dformerv2"), default="legacy")
    parser.add_argument("--tta-small-image-mode", choices=("pad", "resize"), default="pad")
    parser.add_argument("--drop-path-rate", type=float, default=0.10,
                        help="Stochastic depth rate on the encoder (SegFormer regularization). 0.1-0.2 helps against overfitting.")
    parser.add_argument("--prompt-channels", type=int, default=32,
                        help="Width of the compact geometry prompt encoder.")
    parser.add_argument(
        "--fusion-mode",
        type=str,
        default="stagewise",
        choices=("rgb", "prompt", "shallow", "deep", "stagewise", "topology"),
        help="Stage policy used by the clean RSGNet ablation sequence.",
    )
    parser.add_argument(
        "--architecture-variant",
        type=str,
        default="legacy",
        choices=("legacy", "refined", "factorized"),
        help="Select the reproducible A, B, or physics-factorized C model line.",
    )
    parser.add_argument(
        "--geometry-encoding",
        type=str,
        default="factorized_routed",
        choices=(
            "unified",
            "independent",
            "factorized_static",
            "factorized_routed",
            "factorized_swapped",
            "factorized_final",
            "factorized_final_soft",
            "factorized_channel_routed",
        ),
        help="HHA encoding and layout-boundary routing used by the factorized C model line.",
    )
    parser.add_argument(
        "--geometry-channels",
        type=str,
        default="dha",
        choices=("d", "h", "a", "dh", "da", "ha", "dha"),
        help="Canonical HHA channels retained by a capacity-matched ablation.",
    )
    parser.add_argument(
        "--disable-reliability",
        action="store_true",
        help="Disable the reliability head and reliability routing for ablation.",
    )
    parser.add_argument("--decoder-channels", type=int, default=256,
                        help="Width of the fixed lightweight segmentation decoder.")
    parser.add_argument("--attention-tokens", type=int, default=1024,
                        help="Maximum tokens used by deepest-stage semantic attention.")
    parser.add_argument("--local-scale-init", type=float, default=0.05,
                        help="Initial residual scale for shallow fusion and prompt adapters.")
    parser.add_argument("--semantic-scale-init", type=float, default=0.05,
                        help="Initial residual scale for deepest-stage semantic attention.")
    parser.add_argument("--pure-ce", action="store_true", help="Override losses to dense main CE only for collapse diagnosis.")
    parser.add_argument("--lambda-lovasz", type=float, default=0.3)
    parser.add_argument("--lambda-dice", type=float, default=0.0)
    parser.add_argument("--lambda-edge", type=float, default=0.05)
    parser.add_argument("--lambda-boundary", type=float, default=0.0)
    parser.add_argument("--lambda-feature-precision", type=float, default=0.0)
    parser.add_argument("--feature-precision-max-size", type=int, default=160,
                        help="Compute feature precision loss on a downsampled long side to avoid full-resolution memory spikes. Set <=0 to disable downsampling.")
    parser.add_argument("--hha-edge-weight", type=float, default=0.0)
    parser.add_argument("--max-edge-pos-weight", type=float, default=10.0)
    parser.add_argument("--use-class-weights", action="store_true",
                        help="Fold fixed inverse-frequency class weights into CE/boundary loss to help rare classes (mIoU).")
    parser.add_argument("--class-weight-mode", type=str, default="inverse_log", choices=("inverse_log", "inverse_freq"),
                        help="'inverse_log' (gentle, recommended) or 'inverse_freq' (aggressive median-frequency balancing).")
    parser.add_argument("--class-weight-clamp", type=float, default=8.0,
                        help="Upper clamp for per-class weight to keep the loss numerically stable.")
    parser.add_argument("--ohem-min-kept", type=int, default=30000)
    parser.add_argument("--ohem-start-epoch", type=int, default=100000,
                        help="Use dense CE before this 0-based epoch. The stage-wise mainline disables OHEM by default.")
    parser.add_argument("--ce-only-epochs", type=int, default=5, help="Train with dense CE only for the first N epochs.")
    parser.add_argument("--boundary-ce-weight", type=float, default=0.7)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--loss-warmup-start", type=float, default=0.15)
    parser.add_argument("--edge-warmup-epochs", type=int, default=10)
    parser.add_argument("--boundary-warmup-epochs", type=int, default=14)
    parser.add_argument("--feature-warmup-epochs", type=int, default=18)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=15,
                        help="Stop after this many validations without a new best; <=0 disables.")
    parser.add_argument("--early-stopping-min-epoch", type=int, default=30,
                        help="Do not early-stop before this 1-based epoch.")
    parser.add_argument("--encoder-name", type=str, default="mit_b2")
    parser.add_argument("--initial-best-miou", type=float, default=0.0, help="Floor for best_miou when fine-tuning from a weights-only checkpoint.")
    args = parser.parse_args()
    if args.crop_size is not None:
        args.input_height = args.crop_size
        args.input_width = args.crop_size
    apply_protocol(args)
    main(args)
