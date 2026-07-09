import argparse
import os
import warnings

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Harmless DDP perf note triggered by the depthwise conv inside SegFormer's
# MixFFN (the 2048-ch [2048,1,3,3] grad): the size-1 dim makes cuDNN report a
# non-contiguous grad stride that points at the same memory DDP's bucket expects.
# Not an error and does not affect correctness or mIoU; silenced to keep logs clean.
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")

from dataset import INPUT_HEIGHT, INPUT_WIDTH, SUNDataset, get_angle_grad_target
from losses import DGRQCombinedLoss, edge_target_from_mask
from models import UGFLiteNet, remap_dgrq_state_dict
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


CODE_VERSION = "SUNpy-modular-20260607"


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


def save_training_checkpoint(path, epoch, raw_model, ema_model, optimizer, scheduler, best_miou):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "ema_state_dict": get_ema_state_dict(ema_model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_miou": best_miou,
            "code_version": CODE_VERSION,
        },
        path,
    )


def save_best_checkpoints(save_dir, epoch, raw_model, ema_model, optimizer, scheduler, best_miou):
    miou_path = os.path.join(save_dir, f"best_miou_{best_miou:.4f}_epoch_{epoch:03d}.pth")
    save_training_checkpoint(miou_path, epoch, raw_model, ema_model, optimizer, scheduler, best_miou)

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

    rgb_params = collect_params(("encoder.rgb_encoder.",))
    prompt_params = collect_params(
        (
            "encoder.mueb.",
            "encoder.hha_recovery.",
            "encoder.gacr.",
            "encoder.bsde.",
            "encoder.agfd.",
            "encoder.magp.",
            "encoder.fdgpf.",
            "encoder.gstf_layers.",
        )
    )
    fusion_params = collect_params(
        (
            "encoder.dglsr_layers.",
            "encoder.cs_dgam_layers.",
            "encoder.aspp.",
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

    model = UGFLiteNet(
        n_classes=args.n_classes,
        pretrained_path=args.pretrained_encoder,
        return_aux=True,
        encoder_name=args.encoder_name,
        safe_mode=args.safe_mode,
        layout_mode=args.layout_mode,
        decoder_mode=args.decoder_mode,
        drop_path_rate=args.drop_path_rate,
        use_prompt_recovery=not args.no_prompt_recovery,
        prompt_recovery_mode=args.prompt_recovery_mode,
        use_confidence_routing=not args.no_confidence_routing,
        consistency_routing_mode=args.consistency_routing_mode,
        prompt_channels=args.prompt_channels,
        layout_state_dim=args.layout_state_dim,
        geometry_routing_mode=args.geometry_routing_mode,
        reliability_strength=args.reliability_strength,
        decoder_channels=args.decoder_channels,
        attention_tokens=args.attention_tokens,
        local_scale_init=args.local_scale_init,
        semantic_scale_init=args.semantic_scale_init,
        fusion_branch_mode=args.fusion_branch_mode,
        use_prompt_autocorr=not args.no_prompt_autocorr,
        use_directional_edge_refine=not args.no_directional_edge_refine,
        query_points=args.query_points,
        query_scale_init=args.query_scale_init,
        geometry_scale_init=args.geometry_scale_init,
        detail_logit_scale=args.detail_logit_scale,
        rgb_boundary_scale=args.rgb_boundary_scale,
        use_grad_checkpoint=args.use_grad_checkpoint,
    ).to(device)
    if is_main_process():
        layout_aggregator = getattr(model.fdgpf, "layout_aggregator", None)
        layout_backend = getattr(layout_aggregator, "backend", args.layout_mode)
        logger.info(f"Geometry prompt layout mode: {args.layout_mode} | backend: {layout_backend}")
        logger.info(f"Decoder mode: {args.decoder_mode}")
        logger.info(
            "Capacity controls | "
            f"prompt_channels={args.prompt_channels} layout_state_dim={args.layout_state_dim} "
            f"decoder_channels={args.decoder_channels} attention_tokens={args.attention_tokens} "
            f"query_points={args.query_points}"
        )
        logger.info(
            "Geometry routing | "
            f"mode={args.geometry_routing_mode} reliability_strength={args.reliability_strength:g} "
            f"consistency={args.consistency_routing_mode} fusion_branch={args.fusion_branch_mode}"
        )
        logger.info(
            "Residual controls | "
            f"local_scale_init={args.local_scale_init:g} semantic_scale_init={args.semantic_scale_init:g} "
            f"query_scale_init={args.query_scale_init:g} geometry_scale_init={args.geometry_scale_init:g} "
            f"detail_logit_scale={args.detail_logit_scale:g} rgb_boundary_scale={args.rgb_boundary_scale:g}"
        )
        logger.info(
            "Reliable geometry controls | "
            f"prompt_recovery={not args.no_prompt_recovery} "
            f"prompt_recovery_mode={args.prompt_recovery_mode} "
            f"confidence_routing={not args.no_confidence_routing} "
            f"prompt_autocorr={not args.no_prompt_autocorr} "
            f"directional_edge={not args.no_directional_edge_refine}"
        )

    checkpoint = None
    start_epoch = 0
    best_miou = args.initial_best_miou
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location="cpu")
        state_dict = remap_dgrq_state_dict(
            strip_module_prefix(extract_model_state(checkpoint, prefer_ema=args.resume_weights_only))
        )
        state_dict, skipped_mismatch = filter_shape_mismatch(state_dict, model.state_dict())
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if is_main_process():
            logger.info(f"Loaded checkpoint weights from: {args.resume}")
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

    ema_model = build_ema_model(model, decay=args.ema_decay) if args.ema_decay > 0 and is_main_process() else None

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
        hha_name_prefix=args.hha_name_prefix,
        label_name_prefix=args.label_name_prefix,
        ignore_index=args.ignore_index,
        train_cutout_prob=args.train_cutout_prob,
        aug_level=args.aug_level,
    )
    val_dataset = SUNDataset(
        args.data_root,
        dataset_name=args.dataset_name,
        mode="test",
        crop_size=(args.input_height, args.input_width),
        n_classes=args.n_classes,
        label_map_path=args.label_map,
        label_dir_name=args.label_dir_name,
        hha_name_prefix=args.hha_name_prefix,
        label_name_prefix=args.label_name_prefix,
        ignore_index=args.ignore_index,
    )

    train_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=True,
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
        batch_size=args.batch_size,
        shuffle=False,
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

    criterion = DGRQCombinedLoss(
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
        start_epoch = 0 if args.resume_weights_only else checkpoint.get("epoch", 0)
        best_miou = max(checkpoint.get("best_miou", 0.0), args.initial_best_miou)
        if (not args.resume_weights_only) and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if (not args.resume_weights_only) and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if ema_model is not None and checkpoint.get("ema_state_dict") is not None:
            ema_state = remap_dgrq_state_dict(strip_module_prefix(checkpoint["ema_state_dict"]))
            ema_target = ema_model.module if hasattr(ema_model, "module") else ema_model
            ema_state, skipped_ema_mismatch = filter_shape_mismatch(ema_state, ema_target.state_dict())
            ema_target.load_state_dict(ema_state, strict=False)
            if is_main_process() and skipped_ema_mismatch:
                logger.warning(f"Skipped EMA shape-mismatch keys while loading checkpoint: {len(skipped_ema_mismatch)}")
        if is_main_process():
            if args.resume_weights_only:
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
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            ce_only = epoch < args.ce_only_epochs
            use_ohem = epoch >= args.ohem_start_epoch
            curriculum_epoch = max(0, epoch - args.ce_only_epochs)
            edge_loss_scale = 0.0 if ce_only else rampup_factor(curriculum_epoch, args.edge_warmup_epochs, args.loss_warmup_start)
            boundary_loss_scale = 0.0 if ce_only else rampup_factor(curriculum_epoch, args.boundary_warmup_epochs, args.loss_warmup_start)
            feature_loss_scale = 0.0 if ce_only else rampup_factor(curriculum_epoch, args.feature_warmup_epochs, args.loss_warmup_start)

            if is_main_process():
                logger.info(
                    f"Loss curriculum | ce_only={ce_only} ohem={use_ohem} "
                    f"edge={edge_loss_scale:.3f} boundary={boundary_loss_scale:.3f} "
                    f"feature={feature_loss_scale:.3f}"
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

            for step, (rgb, hha, masks) in enumerate(progress):
                rgb = rgb.to(device, non_blocking=True)
                hha = hha.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                masks[(masks != args.ignore_index) & ((masks < 0) | (masks >= args.n_classes))] = args.ignore_index

                valid_edge_mask = (masks != args.ignore_index).float().unsqueeze(1)
                edge_tgt = edge_target_from_mask(masks, ignore_index=args.ignore_index).float()
                angle_grad_tgt = get_angle_grad_target(hha).float() * valid_edge_mask

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    seg_logits, edge_logits, aux_logits = model(rgb, hha)
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
                if is_main_process():
                    use_ema_eval = ema_model is not None and (epoch + 1) >= args.ema_warmup_epochs
                    eval_model = ema_model if use_ema_eval else raw_model
                    val_scales = (0.75, 1.0, 1.25) if args.use_tta else (1.0,)
                    miou = validate(
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
                    )
                    eval_name = "EMA" if use_ema_eval else "raw"
                    logger.info(f"Epoch {epoch + 1} | Validation mIoU ({eval_name}): {miou:.4f}")

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
                        )
                        epochs_without_improvement = 0
                        logger.info(
                            f"New best checkpoint reached: {best_miou:.4f} -> {best_ckpt}"
                        )
                    else:
                        epochs_without_improvement += 1

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
                )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/home/pengfei/HTCnet/DataSets")
    parser.add_argument("--dataset-name", type=str, default="SUNRGBD", help="Dataset folder under --data-root, e.g. SUNRGBD or NYU.")
    parser.add_argument("--save-dir", type=str, default="../Checkpoint_SUN_V8")
    parser.add_argument("--pretrained-encoder", type=str, default="/home/pengfei/HTCnet/Checkpoint/mit_b2.pth")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--resume-weights-only", action="store_true", help="Load checkpoint weights and best mIoU, but reset optimizer, scheduler, and epoch for fine-tuning.")
    parser.add_argument("--label-map", type=str, default="", help='Optional raw-label to train-label map: txt/csv lines "raw train" or JSON dict.')
    parser.add_argument("--label-dir-name", type=str, default="Labels", help="Label folder under --data-root/--dataset-name, e.g. Labels, Labels40, or Label.")
    parser.add_argument("--hha-name-prefix", type=str, default="", help="Optional filename prefix for HHA images, e.g. 'hha_' for NYU.")
    parser.add_argument("--label-name-prefix", type=str, default="", help="Optional filename prefix for label images.")
    parser.add_argument("--train-cutout-prob", type=float, default=0.20,
                        help="Probability of RGB-only cutout during training. Lower it for late fine-tuning.")
    parser.add_argument("--aug-level", type=str, default="strong", choices=("base", "strong"),
                        help="'base' keeps the original augmentation; 'strong' adds hue/brightness/contrast jitter, "
                             "wider scale range, and multi-block RGB+HHA cutout with label voiding.")
    parser.add_argument("--n-classes", type=int, default=41)
    parser.add_argument("--ignore-index", type=int, default=0, help="Void label id ignored by losses and metrics.")
    parser.add_argument("--miou-start-class", type=int, default=1, help="First class id included in mIoU.")
    parser.add_argument("--crop-size", type=int, default=None, help="Deprecated square input size. Prefer --input-height/--input-width.")
    parser.add_argument("--input-height", type=int, default=INPUT_HEIGHT)
    parser.add_argument("--input-width", type=int, default=INPUT_WIDTH)
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
    parser.add_argument("--safe-mode", action="store_true", help="Use convolutional safe-mode instead of angle-guided GSA in the deepest fusion stage.")
    parser.add_argument("--drop-path-rate", type=float, default=0.10,
                        help="Stochastic depth rate on the encoder (SegFormer regularization). 0.1-0.2 helps against overfitting.")
    parser.add_argument("--prompt-channels", type=int, default=32,
                        help="Width of the frequency-aware geometry prompt stream. Increase to 48/64 for a capacity run.")
    parser.add_argument("--layout-state-dim", type=int, default=16,
                        help="State dimension of the existing SSM layout aggregator.")
    parser.add_argument("--geometry-routing-mode", type=str, default="rcfr", choices=("legacy", "rcfr"),
                        help="Geometry prompt/fusion routing variant; rcfr enables reliability-calibrated frequency routing.")
    parser.add_argument("--reliability-strength", type=float, default=0.30,
                        help="Reliability calibration strength for rcfr routing.")
    parser.add_argument("--prompt-recovery-mode", type=str, default="rgpr", choices=("legacy", "rgpr"),
                        help="'legacy' uses HHA-only bounded recovery; 'rgpr' uses RGB-guided geometric prompt recovery.")
    parser.add_argument("--consistency-routing-mode", type=str, default="gacr", choices=("none", "gacr"),
                        help="'gacr' replaces raw confidence routing with geometry-appearance consistency routing.")
    parser.add_argument("--fusion-branch-mode", type=str, default="both", choices=("both", "local", "semantic", "rgb"),
                        help="Fusion branch ablation: use both branches, local only, semantic only, or RGB-only passthrough.")
    parser.add_argument("--no-prompt-autocorr", action="store_true",
                        help="Disable reliability-aware autocorrelation prompt mixing (ablation).")
    parser.add_argument("--no-directional-edge-refine", action="store_true",
                        help="Disable DEGConv-inspired directional geometry edge refinement (ablation).")
    parser.add_argument("--decoder-channels", type=int, default=256,
                        help="Decoder width.")
    parser.add_argument("--decoder-mode", type=str, default="simple_mlp", choices=("qdhs", "simple_mlp"),
                        help="'qdhs' uses the full query-driven decoder; 'simple_mlp' uses a lightweight SegFormer-style decoder ablation.")
    parser.add_argument("--attention-tokens", type=int, default=1024,
                        help="Maximum tokens used by the semantic context branch after adaptive downsampling.")
    parser.add_argument("--use-grad-checkpoint", action="store_true",
                        help="Recompute the attention branch (semantic_attn + deep_gsa) in backward to save activation "
                             "memory. ~20%% slower per step; resolves OOM that hits when the aux losses turn on.")
    parser.add_argument("--local-scale-init", type=float, default=1e-4,
                        help="Initial residual scale for the local fusion branch.")
    parser.add_argument("--semantic-scale-init", type=float, default=1e-4,
                        help="Initial residual scale for the semantic fusion branch.")
    parser.add_argument("--query-points", type=int, default=4,
                        help="Sampling points in the diagnostic multi-scale query attention.")
    parser.add_argument("--query-scale-init", type=float, default=0.05,
                        help="Initial scale of the diagnostic query mask logits.")
    parser.add_argument("--geometry-scale-init", type=float, default=0.1,
                        help="Initial scale of the diagnostic geometry reconstruction logits.")
    parser.add_argument("--detail-logit-scale", type=float, default=0.3,
                        help="Scale for the diagnostic detail-logit residual before boundary refinement.")
    parser.add_argument("--rgb-boundary-scale", type=float, default=0.1,
                        help="Scale for the diagnostic RGB fixed-boundary residual.")
    parser.add_argument("--no-prompt-recovery", action="store_true",
                        help="Disable bounded HHA prompt recovery before geometry prompt construction (ablation).")
    parser.add_argument("--no-confidence-routing", action="store_true",
                        help="Disable reliability-gated geometry routing in the fusion block (ablation).")
    parser.add_argument("--layout-mode", type=str, default="ssm", choices=("ssm", "conv", "avgpool"),
                        help="Geometry prompt layout stream: 'ssm' (Geometry State-Space Layout Scan, proposed), "
                             "'conv' (capacity-matched local baseline), or 'avgpool' (original AvgPool stream).")
    parser.add_argument("--pure-ce", action="store_true", help="Override losses to dense main CE only for collapse diagnosis.")
    parser.add_argument("--lambda-lovasz", type=float, default=0.5)
    parser.add_argument("--lambda-dice", type=float, default=0.4)
    parser.add_argument("--lambda-edge", type=float, default=0.03)
    parser.add_argument("--lambda-boundary", type=float, default=0.01)
    parser.add_argument("--lambda-feature-precision", type=float, default=0.01)
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
    parser.add_argument("--ohem-start-epoch", type=int, default=8, help="Use dense CE before this 0-based epoch, then enable OHEM.")
    parser.add_argument("--ce-only-epochs", type=int, default=5, help="Train with dense CE only for the first N epochs.")
    parser.add_argument("--boundary-ce-weight", type=float, default=0.7)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--loss-warmup-start", type=float, default=0.15)
    parser.add_argument("--edge-warmup-epochs", type=int, default=10)
    parser.add_argument("--boundary-warmup-epochs", type=int, default=14)
    parser.add_argument("--feature-warmup-epochs", type=int, default=18)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--encoder-name", type=str, default="mit_b2")
    parser.add_argument("--initial-best-miou", type=float, default=0.0, help="Floor for best_miou when fine-tuning from a weights-only checkpoint.")
    args = parser.parse_args()
    if args.crop_size is not None:
        args.input_height = args.crop_size
        args.input_width = args.crop_size
    main(args)
