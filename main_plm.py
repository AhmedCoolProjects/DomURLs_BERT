import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import MLFlowLogger
try:
    from lightning.pytorch.loggers import WandbLogger
    _WANDB_AVAILABLE = True
except ImportError:
    WandbLogger = None
    _WANDB_AVAILABLE = False
from torch.utils.data import DataLoader
from data_utils.bertdataset import BERTDataset
from data_utils.load_data import load_dataset
from models.urlbert import URLBertForSequenceClassification
from models.plm import PLMEncoder
from module.pl_module import BaseModel
import yaml
import argparse
import pickle
import warnings
import os
from collections import OrderedDict
from datetime import datetime
from argparse import Namespace
from utils import trace_jit_model
from pathlib import Path


def _compute_class_weights(y_train, num_classes, strategy, cap):
    """Per-class CE weights. Returns None if strategy=='none'.

    Strategies:
      - inverse:      w[c] = N / (K * count[c])   (raw inverse-frequency)
      - sqrt_inverse: w[c] = sqrt(N / count[c]), then normalized to mean 1
      - capped:       inverse-frequency, then clip max weight to cap × min
    """
    if strategy == "none":
        return None
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)            # avoid div-by-zero for absent classes
    N = counts.sum()
    K = num_classes

    if strategy == "inverse":
        w = N / (K * counts)
    elif strategy == "sqrt_inverse":
        w = np.sqrt(N / counts)
        w = w / w.mean()                        # normalize so weights average ~1
    elif strategy == "capped":
        w = N / (K * counts)
        w_min = w.min()
        w = np.minimum(w, cap * w_min)
        w = w / w.mean()                        # re-normalize
    else:
        raise ValueError(f"unknown class_weight_strategy: {strategy}")
    return w.astype(np.float32)


class FocalLoss(torch.nn.Module):
    """Multiclass focal loss. weight=class weights (alpha); gamma down-weights easy ex."""
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = float(gamma)
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce)
        return (((1.0 - p_t) ** self.gamma) * ce).mean()


def main(args):
    # Set random seed
    seed_everything(seed=args.seed)
    # Set dataset path
    path= f'data/{args.experiment_type}_datasets/{args.dataset}'
    # Set the max sequence length
    if args.experiment_type == 'domain':
        max_length = 64
    else:
        max_length = 128

    # Metadata fusion: only meaningful for domain experiments. Auto-disable for URL.
    use_metadata = (not args.no_metadata) and (args.experiment_type == 'domain')
    if (not args.no_metadata) and args.experiment_type != 'domain':
        print("metadata fusion disabled (only supported for --experiment_type domain)")

    meta_categories = None
    if use_metadata:
        if not args.metadata_dir:
            raise ValueError("metadata fusion is enabled but --metadata_dir is empty. "
                             "Pass --no_metadata to disable or provide --metadata_dir.")
        meta_categories = [c.strip() for c in args.meta_categories.split(",") if c.strip()]

    if args.malicious_only and args.label_column == "label":
        raise ValueError("--malicious_only filters to label=='malicious', leaving only one "
                         "class — combine it with --label_column class for stage-2 training.")

    # Load the data
    data_dict = load_dataset(
        path,
        args.label_column,
        meta_dir=args.metadata_dir if use_metadata else None,
        meta_categories=meta_categories,
        malicious_only=args.malicious_only,
    )
    df_train, df_dev, df_test, label_encoder = data_dict['train'], data_dict['dev'], data_dict['test'], data_dict['label_encoder']
    num_classes = len(label_encoder.classes_)

    meta_train = data_dict.get("meta_train")
    meta_dev   = data_dict.get("meta_dev")
    meta_test  = data_dict.get("meta_test")
    meta_dim   = meta_train.shape[1] if meta_train is not None else 0
    if use_metadata:
        print(f"metadata: {meta_dim} features, categories={meta_categories}")
        print(f"meta shapes: train={meta_train.shape} dev={meta_dev.shape} test={meta_test.shape}")
    
    training_params = {"batch_size": args.batch_size,
                        "shuffle": True,
                        "num_workers": args.num_workers,
                        "pin_memory": True,
                        "drop_last":True}
    test_params = {"batch_size": args.batch_size,
                    "shuffle": False,
                    "num_workers": args.num_workers,
                    "pin_memory": True,
                    "drop_last":False}

    model_params = {
        'output_size': num_classes,
        'drop_prob': args.dropout_prob,
        'pretrained_path': args.pretrained_path,
        'meta_dim': meta_dim,
        'meta_hidden_dim': args.meta_hidden_dim,
        'meta_out_dim': args.meta_out_dim,
        'fusion_mode': args.fusion_mode,
        'head_hidden_dim': args.head_hidden_dim,
        'meta_layernorm': args.meta_layernorm,
    }

    experiment_params = {
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    "epochs": args.epochs,
    "gradient_clip_val": 0.9,
    "dataset" : args.dataset,
    'pretrained_path': args.pretrained_path,
    "max_length": max_length,
    "num_workers" : args.num_workers,
    "drop_prob": args.dropout_prob,
    "batch_size": args.batch_size,
    "experiment_type": args.experiment_type,
    "label_column" : args.label_column,
    "num_classes": num_classes,
    "use_metadata": use_metadata,
    "meta_dim": meta_dim,
    "meta_categories": meta_categories,
    "meta_hidden_dim": args.meta_hidden_dim,
    "meta_out_dim": args.meta_out_dim,
    "fusion_mode": args.fusion_mode,
    "head_hidden_dim": args.head_hidden_dim,
    "meta_layernorm": args.meta_layernorm,
    "class_weight_strategy": args.class_weight_strategy,
    "class_weight_cap": args.class_weight_cap,
    "focal_loss": args.focal_loss,
    "focal_gamma": args.focal_gamma,
    "malicious_only": args.malicious_only,
    }

    config = Namespace(**experiment_params)

    train_dataset = BERTDataset(texts=df_train['input'].values, labels=df_train[args.label_column].values, max_length=max_length, pretrained_path=args.pretrained_path, meta_vectors=meta_train)
    dev_dataset = BERTDataset(texts=df_dev['input'].values, labels=df_dev[args.label_column].values, max_length=max_length, pretrained_path=args.pretrained_path, meta_vectors=meta_dev)
    test_dataset = BERTDataset(texts=df_test['input'].values, labels=df_test[args.label_column].values, max_length=max_length, pretrained_path=args.pretrained_path, meta_vectors=meta_test)
    
    train_loader = DataLoader(train_dataset, **training_params)
    dev_loader = DataLoader(dev_dataset, **test_params)
    test_loader = DataLoader(test_dataset, **test_params)

    #MLFlow Experiment tracking setup

    experiment_tags = OrderedDict({
    "project_name": "DomURLs_BERT",
    "Domains_labels": ", ".join(label_encoder.classes_),
    "mlflow.note.content":f"Deep learning for malicious {args.experiment_type.upper()} detection and classification using {args.pretrained_path} model",
    })

    mlflow.set_tracking_uri(Path.cwd().joinpath("mlruns/experiments").as_uri())
    exp_name = f'{args.dataset}_binary_cls_{num_classes==2}_{args.experiment_type}'
    print('experiment', exp_name)
    exp_id = mlflow.set_experiment(exp_name)
    mlflow.pytorch.autolog()
    run_name = f"{args.pretrained_path.split('/')[1]}" if '/' in args.pretrained_path else args.pretrained_path
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(experiment_params)
    mlflow.set_tags(experiment_tags)


    mlf_logger = MLFlowLogger(
        experiment_name=mlflow.get_experiment(mlflow.active_run().info.experiment_id).name,
        tracking_uri=mlflow.get_tracking_uri(),
        run_id=mlflow.active_run().info.run_id,
        log_model = False
        )
    experiment_id = mlflow.active_run().info.experiment_id
    run_id = mlflow.active_run().info.run_id

    # --- Optional wandb logger (alongside mlflow) ---
    loggers = [mlf_logger]
    use_wandb = (not args.no_wandb) and _WANDB_AVAILABLE
    if args.no_wandb:
        print("wandb disabled by --no_wandb")
    elif not _WANDB_AVAILABLE:
        print("wandb not installed; skipping wandb logging")
    if use_wandb:
        model_short = args.pretrained_path.split('/')[-1]
        meta_tag = "no_meta" if not use_metadata else f"meta_{','.join(meta_categories) if meta_categories else 'all'}"
        wandb_run_name = args.wandb_run_name or f"{model_short}__{meta_tag}__{args.dataset}__{args.label_column}"
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=wandb_run_name,
            tags=["plm", args.dataset, args.label_column,
                  args.pretrained_path, "metadata" if use_metadata else "no_metadata"],
            log_model=False,
            config=experiment_params,
        )
        loggers.append(wandb_logger)

    ckpts_path = './mlruns/ckpts'
    checkpoint_path = f"{ckpts_path}/{run_id}/"
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, monitor='val_loss',  
                                            filename=f'best_checkpoint')
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=5)

    # Instanciate the model, loss, and the Lightning module
    
    model = None
    if 'urlbert' in args.pretrained_path.lower():
        classifier = URLBertForSequenceClassification(**model_params)
    else:
        classifier = PLMEncoder(**model_params)

    # Build the loss: optional class weights + optional focal modulation.
    weights_np = _compute_class_weights(
        df_train[args.label_column].to_numpy(),
        num_classes,
        args.class_weight_strategy,
        args.class_weight_cap,
    )
    weights_t = torch.tensor(weights_np, dtype=torch.float32) if weights_np is not None else None
    if weights_t is not None:
        print(f"class weights ({args.class_weight_strategy}): "
              f"min={weights_t.min().item():.3f}  max={weights_t.max().item():.3f}  "
              f"mean={weights_t.mean().item():.3f}")

    if args.focal_loss:
        criterion = FocalLoss(gamma=args.focal_gamma, weight=weights_t)
        print(f"FocalLoss(gamma={args.focal_gamma}, weighted={weights_t is not None})")
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=weights_t)

    lit_model = BaseModel(classifier=classifier, num_classes=num_classes, criterion=criterion, config=config, names = label_encoder.classes_)

    # Instanciate model trainer
    trainer = Trainer(max_epochs=args.epochs, devices=[args.device], accelerator="gpu", logger=loggers, callbacks=[checkpoint_callback, early_stopping],  benchmark=False,
                  deterministic=True, precision="16", gradient_clip_val=1)
    # Model training
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=dev_loader)
    # Evaluate the model on the dev set
    dev_results =  trainer.validate(ckpt_path='best', dataloaders=dev_loader)
    # Evaluate the model on the test set
    test_results = trainer.test(ckpt_path='best', dataloaders=test_loader)

    # Logging experiment metadata artifact
    path_metadata = f"{checkpoint_path}/model_metadata.p"
    experiment_params['label_encoder'] = label_encoder
    if use_metadata:
        experiment_params['meta_scaler']        = data_dict.get('meta_scaler')
        experiment_params['meta_feature_cols']  = data_dict.get('meta_feature_cols')
    with open(path_metadata, 'wb') as f:
        pickle.dump(experiment_params, f)
    mlflow.log_artifact(path_metadata, artifact_path=f"model_metadata.p")

    # Logging classification report artifact
    test_report_path = f"{checkpoint_path}/test_classification_report.txt"
    with open(test_report_path, 'w') as f:
        f.write("Classification report----------------------------------\n\n")
        f.write(lit_model.classification_report)
    mlflow.log_artifact(test_report_path, artifact_path=f"classification_report")
    

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description="Train PLM-based models for malicious URL and domain names detection.")
    parser.add_argument('--dataset', type=str, default='Mendeley_AK_Singh_2020_phish', help='Dataset name')
    parser.add_argument('--pretrained_path', type=str, default='amahdaouy/DomURLs_BERT', help='Pretrained model path')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for data loading')
    parser.add_argument('--dropout_prob', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--experiment_type', type=str, default='url', choices=["url", "domain"], help='Type of experiment')
    parser.add_argument('--label_column', type=str, default='label', help='Name of the label column')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')
    parser.add_argument('--device', type=int, default=0, help='GPU device id')
    # Metadata fusion (only for --experiment_type domain). Default ON; opt-out with --no_metadata.
    parser.add_argument('--no_metadata', action='store_true',
                        help='Disable metadata fusion (default: enabled).')
    parser.add_argument('--metadata_dir', type=str,
                        default='/home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/domains/DomainsMetadata',
                        help='Directory containing {train,dev,test}_meta.csv')
    parser.add_argument('--meta_categories', type=str, default='rdap,dns,tls,ip',
                        help='Comma-separated metadata category prefixes to include '
                             '(default: drop_x_ct, the round-2 ablation winner).')
    parser.add_argument('--meta_hidden_dim', type=int, default=128,
                        help='Hidden dim of the metadata MLP')
    parser.add_argument('--meta_out_dim', type=int, default=64,
                        help='Output dim of the metadata MLP (concatenated to [CLS])')
    parser.add_argument('--fusion_mode', type=str, default='concat', choices=['concat', 'gated'],
                        help='How to fuse meta with [CLS]. "gated" applies a learned per-sample '
                             'sigmoid gate to MetaMLP(meta) before concatenation.')
    parser.add_argument('--head_hidden_dim', type=int, default=0,
                        help='If > 0, replace the final Linear classifier with a 2-layer MLP head '
                             'of this hidden width (Linear -> GELU -> Dropout -> Linear).')
    parser.add_argument('--meta_layernorm', action='store_true',
                        help='Apply LayerNorm to MetaMLP output before fusion.')
    parser.add_argument('--class_weight_strategy', type=str, default='none',
                        choices=['none', 'inverse', 'sqrt_inverse', 'capped'],
                        help='Per-class weighting in the loss. "inverse" is raw 1/freq '
                             '(can destabilize); "sqrt_inverse" is gentler; "capped" caps '
                             'the max/min ratio at --class_weight_cap.')
    parser.add_argument('--class_weight_cap', type=float, default=10.0,
                        help='Max weight / min weight ratio when --class_weight_strategy=capped.')
    parser.add_argument('--focal_loss', action='store_true',
                        help='Use focal loss instead of CE. Combines with --class_weight_strategy.')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma. 0 reduces to weighted CE.')
    parser.add_argument('--malicious_only', action='store_true',
                        help='Stage-2 of two-stage classifier: drop legit rows before '
                             'encoding labels, train only on malicious. Use with '
                             '--label_column class.')
    # wandb (logged alongside mlflow)
    parser.add_argument('--wandb_project', type=str, default='DomURLs_BERT_metadata',
                        help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='wandb entity (team/user); defaults to your wandb login')
    parser.add_argument('--wandb_group', type=str, default=None,
                        help='wandb group name; useful to bucket related runs')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='wandb run name; default derived from model+meta+dataset')
    parser.add_argument('--no_wandb', action='store_true', help='disable wandb logging')
    args = parser.parse_args()

    main(args=args)