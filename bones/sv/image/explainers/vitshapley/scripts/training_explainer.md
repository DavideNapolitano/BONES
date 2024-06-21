# Explainer

## Masking method - Pre-softmax

### ImageNette

```bash
python main.py with 'stage = "explainer"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "ImageNette_vit_tiny_patch16_224_explainer_lr1e-4_additive_block1_clstrue_mlp3_ratio4_freezefalse_normtrue_softmax_acttanh"' \
env_chanwkim 'gpus_surrogate=[2]' 'gpus_explainer=[3]' \
dataset_ImageNette \
'surrogate_mask_location = "pre-softmax"' \
'surrogate_backbone_type = "vit_tiny_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/wandb_transformer_interpretability_project/1wyk4vw3/checkpoints/epoch=39-step=5879.ckpt"' \
'explainer_num_mask_samples = 32' 'explainer_num_mask_samples_epoch = 0' 'explainer_paired_mask_samples = True' \
'explainer_normalization = "additive"' 'explainer_normalization_class = None' 'explainer_link = "softmax"' 'explainer_head_num_attention_blocks = 1' 'explainer_head_include_cls = True' 'explainer_head_num_mlp_layers = 3' 'explainer_norm = True' 'explainer_freeze_backbone = "none"' 'explainer_head_mlp_layer_ratio = 4' 'explainer_activation="tanh"' \
'explainer_backbone_type = "vit_tiny_patch16_224"' 'explainer_download_weight = False' 'explainer_load_path = "results/wandb_transformer_interpretability_project/1wyk4vw3/checkpoints/epoch=39-step=5879.ckpt"' \
'unfreeze_after=None' 'unfreeze_after_gradual = False' \
training_hyperparameters_transformer 'per_gpu_batch_size = 32' 'checkpoint_metric = "loss"' 'learning_rate = 1e-4' 'max_epochs = 100'
```

```bash
python main.py with 'stage = "explainer"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "ImageNette_vit_small_patch16_224_explainer_lr1e-4_additive_block1_clstrue_mlp3_ratio4_freezefalse_normtrue_softmax_acttanh"' \
env_chanwkim 'gpus_surrogate=[2]' 'gpus_explainer=[3]' \
dataset_ImageNette \
'surrogate_mask_location = "pre-softmax"' \
'surrogate_backbone_type = "vit_small_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/wandb_transformer_interpretability_project/3lfv4nmn/checkpoints/epoch=39-step=5879.ckpt"' \
'explainer_num_mask_samples = 32' 'explainer_num_mask_samples_epoch = 0' 'explainer_paired_mask_samples = True' \
'explainer_normalization = "additive"' 'explainer_normalization_class = None' 'explainer_link = "softmax"' 'explainer_head_num_attention_blocks = 1' 'explainer_head_include_cls = True' 'explainer_head_num_mlp_layers = 3' 'explainer_norm = True' 'explainer_freeze_backbone = "none"' 'explainer_head_mlp_layer_ratio = 4' 'explainer_activation="tanh"' \
'explainer_backbone_type = "vit_small_patch16_224"' 'explainer_download_weight = False' 'explainer_load_path = "results/wandb_transformer_interpretability_project/3lfv4nmn/checkpoints/epoch=39-step=5879.ckpt"' \
'unfreeze_after=None' 'unfreeze_after_gradual = False' \
training_hyperparameters_transformer 'per_gpu_batch_size = 32' 'checkpoint_metric = "loss"' 'learning_rate = 1e-4' 'max_epochs = 100'
```

```bash
python main.py with 'stage = "explainer"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "ImageNette_vit_base_patch16_224_explainer_lr1e-4_additive_block1_clstrue_mlp3_ratio4_freezefalse_normtrue_softmax_acttanh"' \
env_chanwkim 'gpus_surrogate=[2]' 'gpus_explainer=[3]' \
dataset_ImageNette \
'surrogate_mask_location = "pre-softmax"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/wandb_transformer_interpretability_project/7y487b5g/checkpoints/epoch=38-step=5732.ckpt"' \
'explainer_num_mask_samples = 32' 'explainer_num_mask_samples_epoch = 0' 'explainer_paired_mask_samples = True' \
'explainer_normalization = "additive"' 'explainer_normalization_class = None' 'explainer_link = "softmax"' 'explainer_head_num_attention_blocks = 1' 'explainer_head_include_cls = True' 'explainer_head_num_mlp_layers = 3' 'explainer_norm = True' 'explainer_freeze_backbone = "none"' 'explainer_head_mlp_layer_ratio = 4' 'explainer_activation="tanh"' \
'explainer_backbone_type = "vit_base_patch16_224"' 'explainer_download_weight = False' 'explainer_load_path = "results/wandb_transformer_interpretability_project/7y487b5g/checkpoints/epoch=38-step=5732.ckpt"' \
'unfreeze_after=None' 'unfreeze_after_gradual = False' \
training_hyperparameters_transformer 'per_gpu_batch_size = 32' 'checkpoint_metric = "loss"' 'learning_rate = 1e-4' 'max_epochs = 100'
```

```bash
python main-Copy1.py with 'stage = "explainer"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "ImageNette_vit_large_patch16_224_explainer_lr1e-4_additive_block1_clstrue_mlp3_ratio4_freezefalse_normtrue_softmax_acttanh"' \
env_chanwkim 'gpus_surrogate=[0]' 'gpus_explainer=[1]' \
dataset_ImageNette \
'surrogate_mask_location = "pre-softmax"' \
'surrogate_backbone_type = "vit_large_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/wandb_transformer_interpretability_project/284sm0on/checkpoints/epoch=37-step=5585.ckpt"' \
'explainer_num_mask_samples = 32' 'explainer_num_mask_samples_epoch = 0' 'explainer_paired_mask_samples = True' \
'explainer_normalization = "additive"' 'explainer_normalization_class = None' 'explainer_link = "softmax"' 'explainer_head_num_attention_blocks = 1' 'explainer_head_include_cls = True' 'explainer_head_num_mlp_layers = 3' 'explainer_norm = True' 'explainer_freeze_backbone = "none"' 'explainer_head_mlp_layer_ratio = 4' 'explainer_activation="tanh"' \
'explainer_backbone_type = "vit_large_patch16_224"' 'explainer_download_weight = False' 'explainer_load_path = "results/wandb_transformer_interpretability_project/284sm0on/checkpoints/epoch=37-step=5585.ckpt"' \
'unfreeze_after=None' 'unfreeze_after_gradual = False' \
training_hyperparameters_transformer 'per_gpu_batch_size = 32' 'checkpoint_metric = "loss"' 'learning_rate = 1e-4' 'max_epochs = 100'  'precision = 32'
```



### hyperparameter choices

```bash
python main.py with 'stage = "explainer"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "ImageNette_vit_base_patch16_224_explainer_lr1e-4_additive_block2_clstrue_mlp3_ratio4_freezefalse_normtrue_softmax_acttanh"' \
env_chanwkim 'gpus_surrogate=[4]' 'gpus_explainer=[5]' \
dataset_ImageNette \
'surrogate_mask_location = "pre-softmax"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/wandb_transformer_interpretability_project/7y487b5g/checkpoints/epoch=38-step=5732.ckpt"' \
'explainer_num_mask_samples = 32' 'explainer_num_mask_samples_epoch = 0' 'explainer_paired_mask_samples = True' \
'explainer_normalization = "additive"' 'explainer_normalization_class = None' 'explainer_link = "softmax"' 'explainer_head_num_attention_blocks = 2' 'explainer_head_include_cls = True' 'explainer_head_num_mlp_layers = 3' 'explainer_norm = True' 'explainer_freeze_backbone = "none"' 'explainer_head_mlp_layer_ratio = 4' 'explainer_activation="tanh"' \
'explainer_backbone_type = "vit_base_patch16_224"' 'explainer_download_weight = False' 'explainer_load_path = "results/wandb_transformer_interpretability_project/7y487b5g/checkpoints/epoch=38-step=5732.ckpt"' \
'unfreeze_after=None' 'unfreeze_after_gradual = False' \
training_hyperparameters_transformer 'per_gpu_batch_size = 32' 'checkpoint_metric = "loss"' 'learning_rate = 1e-4' 'max_epochs = 100'
```

```bash
python main.py with 'stage = "explainer"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "ImageNette_vit_base_patch16_224_explainer_lr1e-4_additive_block1_clstrue_mlp3_ratio4_freezetrue_normtrue_softmax_acttanh"' \
env_chanwkim 'gpus_surrogate=[1]' 'gpus_explainer=[2]' \
dataset_ImageNette \
'surrogate_mask_location = "pre-softmax"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/wandb_transformer_interpretability_project/7y487b5g/checkpoints/epoch=38-step=5732.ckpt"' \
'explainer_num_mask_samples = 32' 'explainer_num_mask_samples_epoch = 0' 'explainer_paired_mask_samples = True' \
'explainer_normalization = "additive"' 'explainer_normalization_class = None' 'explainer_link = "softmax"' 'explainer_head_num_attention_blocks = 1' 'explainer_head_include_cls = True' 'explainer_head_num_mlp_layers = 3' 'explainer_norm = True' 'explainer_freeze_backbone = "all"' 'explainer_head_mlp_layer_ratio = 4' 'explainer_activation="tanh"' \
'explainer_backbone_type = "vit_base_patch16_224"' 'explainer_download_weight = False' 'explainer_load_path = "results/wandb_transformer_interpretability_project/7y487b5g/checkpoints/epoch=38-step=5732.ckpt"' \
'unfreeze_after=None' 'unfreeze_after_gradual = False' \
training_hyperparameters_transformer 'per_gpu_batch_size = 16' 'checkpoint_metric = "loss"' 'learning_rate = 1e-4' 'max_epochs = 100'
```

```bash
python main.py with 'stage = "explainer"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "ImageNette_vit_base_patch16_224_explainer_lr1e-4_additive_block0_clstrue_mlp3_ratio4_freezefalse_normtrue_softmax_acttanh"' \
env_chanwkim 'gpus_surrogate=[3]' 'gpus_explainer=[4]' \
dataset_ImageNette \
'surrogate_mask_location = "pre-softmax"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/wandb_transformer_interpretability_project/7y487b5g/checkpoints/epoch=38-step=5732.ckpt"' \
'explainer_num_mask_samples = 32' 'explainer_num_mask_samples_epoch = 0' 'explainer_paired_mask_samples = True' \
'explainer_normalization = "additive"' 'explainer_normalization_class = None' 'explainer_link = "softmax"' 'explainer_head_num_attention_blocks = 0' 'explainer_head_include_cls = True' 'explainer_head_num_mlp_layers = 3' 'explainer_norm = True' 'explainer_freeze_backbone = "none"' 'explainer_head_mlp_layer_ratio = 4' 'explainer_activation="tanh"' \
'explainer_backbone_type = "vit_base_patch16_224"' 'explainer_download_weight = False' 'explainer_load_path = "results/wandb_transformer_interpretability_project/7y487b5g/checkpoints/epoch=38-step=5732.ckpt"' \
'unfreeze_after=None' 'unfreeze_after_gradual = False' \
training_hyperparameters_transformer 'per_gpu_batch_size = 16' 'checkpoint_metric = "loss"' 'learning_rate = 1e-4' 'max_epochs = 100'
```

```bash
python main.py with 'stage = "explainer"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "ImageNette_vit_base_patch16_224_explainer_lr1e-4_additive_block0_clstrue_mlp3_ratio4_freezetrue_normtrue_softmax_acttanh"' \
env_chanwkim 'gpus_surrogate=[2]' 'gpus_explainer=[4]' \
dataset_ImageNette \
'surrogate_mask_location = "pre-softmax"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/wandb_transformer_interpretability_project/7y487b5g/checkpoints/epoch=38-step=5732.ckpt"' \
'explainer_num_mask_samples = 32' 'explainer_num_mask_samples_epoch = 0' 'explainer_paired_mask_samples = True' \
'explainer_normalization = "additive"' 'explainer_normalization_class = None' 'explainer_link = "softmax"' 'explainer_head_num_attention_blocks = 0' 'explainer_head_include_cls = True' 'explainer_head_num_mlp_layers = 3' 'explainer_norm = True' 'explainer_freeze_backbone = "all"' 'explainer_head_mlp_layer_ratio = 4' 'explainer_activation="tanh"' \
'explainer_backbone_type = "vit_base_patch16_224"' 'explainer_download_weight = False' 'explainer_load_path = "results/wandb_transformer_interpretability_project/7y487b5g/checkpoints/epoch=38-step=5732.ckpt"' \
'unfreeze_after=None' 'unfreeze_after_gradual = False' \
training_hyperparameters_transformer 'per_gpu_batch_size = 16' 'checkpoint_metric = "loss"' 'learning_rate = 1e-4' 'max_epochs = 100'
```

### MURA

```bash
python main.py with 'stage = "explainer"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "MURA_vit_base_patch16_224_explainer_lr1e-4_additive_block1_clstrue_mlp3_ratio4_freezefalse_normtrue_sigmoid_acttanh"' \
env_chanwkim 'gpus_surrogate=[0]' 'gpus_explainer=[3]' \
dataset_MURA \
'surrogate_mask_location = "pre-softmax"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/wandb_transformer_interpretability_project/22ompjqu/checkpoints/epoch=47-step=24767.ckpt"' \
'explainer_num_mask_samples = 32' 'explainer_num_mask_samples_epoch = 0' 'explainer_paired_mask_samples = True' \
'explainer_normalization = "additive"' 'explainer_normalization_class = None' 'explainer_link = "sigmoid"' 'explainer_head_num_attention_blocks = 1' 'explainer_head_include_cls = True' 'explainer_head_num_mlp_layers = 3' 'explainer_norm = True' 'explainer_freeze_backbone = "none"' 'explainer_head_mlp_layer_ratio = 4' 'explainer_activation="tanh"' \
'explainer_backbone_type = "vit_base_patch16_224"' 'explainer_download_weight = False' 'explainer_load_path = "results/wandb_transformer_interpretability_project/22ompjqu/checkpoints/epoch=47-step=24767.ckpt"' \
'unfreeze_after=None' 'unfreeze_after_gradual = False' \
training_hyperparameters_transformer 'per_gpu_batch_size = 32' 'checkpoint_metric = "loss"' 'learning_rate = 1e-4' 'max_epochs = 100'
```

### Pet

```bash
python main.py with 'stage = "explainer"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "Pet_vit_base_patch16_224_explainer_lr1e-4_additive_block1_clstrue_mlp3_ratio4_freezefalse_normtrue_softmax_acttanh"' \
env_chanwkim 'gpus_surrogate=[4]' 'gpus_explainer=[5]' \
dataset_Pet \
'surrogate_mask_location = "pre-softmax"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/wandb_transformer_interpretability_project/146vf465/checkpoints/epoch=40-step=3730.ckpt"' \
'explainer_num_mask_samples = 32' 'explainer_num_mask_samples_epoch = 0' 'explainer_paired_mask_samples = True' \
'explainer_normalization = "additive"' 'explainer_normalization_class = None' 'explainer_link = "softmax"' 'explainer_head_num_attention_blocks = 1' 'explainer_head_include_cls = True' 'explainer_head_num_mlp_layers = 3' 'explainer_norm = True' 'explainer_freeze_backbone = "none"' 'explainer_head_mlp_layer_ratio = 4' 'explainer_activation="tanh"' \
'explainer_backbone_type = "vit_base_patch16_224"' 'explainer_download_weight = False' 'explainer_load_path = "results/wandb_transformer_interpretability_project/146vf465/checkpoints/epoch=40-step=3730.ckpt"' \
'unfreeze_after=None' 'unfreeze_after_gradual = False' \
training_hyperparameters_transformer 'per_gpu_batch_size = 32' 'checkpoint_metric = "loss"' 'learning_rate = 1e-4' 'max_epochs = 100'
```

## Masking method - Zero-input

### ImageNette

```bash
python main.py with 'stage = "explainer"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "ImageNette_vit_base_patch16_224_explainer_lr1e-4_additive_block1_clstrue_mlp3_ratio4_freezefalse_normtrue_softmax_acttanh_zeroinput"' \
env_chanwkim 'gpus_surrogate=[4]' 'gpus_explainer=[5]' \
dataset_ImageNette \
'surrogate_mask_location = "zero-input"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/wandb_transformer_interpretability_project/zyybgzcm/checkpoints/epoch=22-step=3380.ckpt"' \
'explainer_num_mask_samples = 32' 'explainer_num_mask_samples_epoch = 0' 'explainer_paired_mask_samples = True' \
'explainer_normalization = "additive"' 'explainer_normalization_class = None' 'explainer_link = "softmax"' 'explainer_head_num_attention_blocks = 1' 'explainer_head_include_cls = True' 'explainer_head_num_mlp_layers = 3' 'explainer_norm = True' 'explainer_freeze_backbone = "none"' 'explainer_head_mlp_layer_ratio = 4' 'explainer_activation="tanh"' \
'explainer_backbone_type = "vit_base_patch16_224"' 'explainer_download_weight = False' 'explainer_load_path = "results/wandb_transformer_interpretability_project/zyybgzcm/checkpoints/epoch=22-step=3380.ckpt"' \
'unfreeze_after=None' 'unfreeze_after_gradual = False' \
training_hyperparameters_transformer 'per_gpu_batch_size = 32' 'checkpoint_metric = "loss"' 'learning_rate = 1e-4' 'max_epochs = 100'
```

### MURA

```bash
python main.py with 'stage = "explainer"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "MURA_vit_base_patch16_224_explainer_lr1e-4_additive_block1_clstrue_mlp3_ratio4_freezefalse_normtrue_sigmoid_acttanh_zeroinput"' \
env_chanwkim 'gpus_surrogate=[2]' 'gpus_explainer=[3]' \
dataset_MURA \
'surrogate_mask_location = "zero-input"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/wandb_transformer_interpretability_project/2z2qs6t0/checkpoints/epoch=44-step=23219.ckpt"' \
'explainer_num_mask_samples = 32' 'explainer_num_mask_samples_epoch = 0' 'explainer_paired_mask_samples = True' \
'explainer_normalization = "additive"' 'explainer_normalization_class = None' 'explainer_link = "sigmoid"' 'explainer_head_num_attention_blocks = 1' 'explainer_head_include_cls = True' 'explainer_head_num_mlp_layers = 3' 'explainer_norm = True' 'explainer_freeze_backbone = "none"' 'explainer_head_mlp_layer_ratio = 4' 'explainer_activation="tanh"' \
'explainer_backbone_type = "vit_base_patch16_224"' 'explainer_download_weight = False' 'explainer_load_path = "results/wandb_transformer_interpretability_project/2z2qs6t0/checkpoints/epoch=44-step=23219.ckpt"' \
'unfreeze_after=None' 'unfreeze_after_gradual = False' \
training_hyperparameters_transformer 'per_gpu_batch_size = 32' 'checkpoint_metric = "loss"' 'learning_rate = 1e-4' 'max_epochs = 100'
```

### Test

```bash
python main.py with 'stage = "explainer"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "ImageNette_vit_base_patch16_224_explainer_lr1e-4_additive_block1_clstrue_mlp3_ratio4_freezefalse_normtrue_softmax_acttanh_test"' \
env_chanwkim 'gpus_surrogate=[2]' 'gpus_explainer=[3]' \
dataset_ImageNette \
'surrogate_mask_location = "pre-softmax"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/wandb_transformer_interpretability_project/7y487b5g/checkpoints/epoch=38-step=5732.ckpt"' \
'explainer_num_mask_samples = 32' 'explainer_num_mask_samples_epoch = 0' 'explainer_paired_mask_samples = True' \
'explainer_normalization = "additive"' 'explainer_normalization_class = None' 'explainer_link = "softmax"' 'explainer_head_num_attention_blocks = 1' 'explainer_head_include_cls = True' 'explainer_head_num_mlp_layers = 3' 'explainer_norm = True' 'explainer_freeze_backbone = "none"' 'explainer_head_mlp_layer_ratio = 4' 'explainer_activation="tanh"' \
'explainer_backbone_type = "vit_base_patch16_224"' 'explainer_download_weight = False' 'explainer_load_path = "results/wandb_transformer_interpretability_project/3ty85eft/checkpoints/epoch=83-step=12431.ckpt"' \
'test_only = True' training_hyperparameters_transformer 'per_gpu_batch_size = 32' 'checkpoint_metric = "loss"'
```

```bash
python main.py with 'stage = "explainer"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "ImageNette_vit_base_patch16_224_explainer_lr1e-4_additive_block1_clstrue_mlp3_ratio4_freezefalse_normtrue_softmax_acttanh_test_zeroweight"' \
env_chanwkim 'gpus_surrogate=[2]' 'gpus_explainer=[3]' \
dataset_ImageNette \
'surrogate_mask_location = "pre-softmax"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/wandb_transformer_interpretability_project/7y487b5g/checkpoints/epoch=38-step=5732.ckpt"' \
'explainer_num_mask_samples = 32' 'explainer_num_mask_samples_epoch = 0' 'explainer_paired_mask_samples = True' \
'explainer_normalization = "additive"' 'explainer_normalization_class = None' 'explainer_link = "softmax"' 'explainer_head_num_attention_blocks = 1' 'explainer_head_include_cls = True' 'explainer_head_num_mlp_layers = 3' 'explainer_norm = True' 'explainer_freeze_backbone = "none"' 'explainer_head_mlp_layer_ratio = 4' 'explainer_activation="tanh"' \
'explainer_backbone_type = "vit_base_patch16_224"' 'explainer_download_weight = False' 'explainer_load_path = "results/wandb_transformer_interpretability_project/24ffapav/checkpoints/epoch=96-step=14355.ckpt"' \
'test_only = True' training_hyperparameters_transformer 'per_gpu_batch_size = 32' 'checkpoint_metric = "loss"'
```

```bash
python main.py with 'stage = "explainer"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "ImageNette_vit_base_patch16_224_explainer_lr1e-4_additive_block1_clstrue_mlp3_ratio4_freezefalse_normtrue_softmax_acttanh_zeroinput_test"' \
env_chanwkim 'gpus_surrogate=[4]' 'gpus_explainer=[5]' \
dataset_ImageNette \
'surrogate_mask_location = "zero-input"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/wandb_transformer_interpretability_project/zyybgzcm/checkpoints/epoch=22-step=3380.ckpt"' \
'explainer_num_mask_samples = 32' 'explainer_num_mask_samples_epoch = 0' 'explainer_paired_mask_samples = True' \
'explainer_normalization = "additive"' 'explainer_normalization_class = None' 'explainer_link = "softmax"' 'explainer_head_num_attention_blocks = 1' 'explainer_head_include_cls = True' 'explainer_head_num_mlp_layers = 3' 'explainer_norm = True' 'explainer_freeze_backbone = "none"' 'explainer_head_mlp_layer_ratio = 4' 'explainer_activation="tanh"' \
'explainer_backbone_type = "vit_base_patch16_224"' 'explainer_download_weight = False' 'explainer_load_path = "results/wandb_transformer_interpretability_project/24ffapav/checkpoints/epoch=96-step=14355.ckpt"' \
'test_only = True' training_hyperparameters_transformer 'per_gpu_batch_size = 32' 'checkpoint_metric = "loss"'
```

```bash
python main.py with 'stage = "explainer"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "ImageNette_vit_base_patch16_224_explainer_lr1e-4_additive_block1_clstrue_mlp3_ratio4_freezefalse_normtrue_softmax_acttanh_zeroinput_test_presoftmaxweight"' \
env_chanwkim 'gpus_surrogate=[4]' 'gpus_explainer=[5]' \
dataset_ImageNette \
'surrogate_mask_location = "zero-input"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/wandb_transformer_interpretability_project/zyybgzcm/checkpoints/epoch=22-step=3380.ckpt"' \
'explainer_num_mask_samples = 32' 'explainer_num_mask_samples_epoch = 0' 'explainer_paired_mask_samples = True' \
'explainer_normalization = "additive"' 'explainer_normalization_class = None' 'explainer_link = "softmax"' 'explainer_head_num_attention_blocks = 1' 'explainer_head_include_cls = True' 'explainer_head_num_mlp_layers = 3' 'explainer_norm = True' 'explainer_freeze_backbone = "none"' 'explainer_head_mlp_layer_ratio = 4' 'explainer_activation="tanh"' \
'explainer_backbone_type = "vit_base_patch16_224"' 'explainer_download_weight = False' 'explainer_load_path = "results/wandb_transformer_interpretability_project/3ty85eft/checkpoints/epoch=83-step=12431.ckpt"' \
'test_only = True' training_hyperparameters_transformer 'per_gpu_batch_size = 32' 'checkpoint_metric = "loss"'
```

## Masking method - Zero-input-Classifier-Masked

```bash
python main.py with 'stage = "explainer"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "ImageNette_vit_base_patch16_224_explainer_lr1e-4_additive_block1_clstrue_mlp3_ratio4_freezefalse_normtrue_softmax_acttanh_zeroinput_classifiermasked"' \
env_chanwkim 'gpus_surrogate=[3]' 'gpus_explainer=[4]' \
dataset_ImageNette \
'surrogate_mask_location = "zero-input"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/wandb_transformer_interpretability_project/1ytqhm00/checkpoints/epoch=22-step=3380.ckpt"' \
'explainer_num_mask_samples = 32' 'explainer_num_mask_samples_epoch = 0' 'explainer_paired_mask_samples = True' \
'explainer_normalization = "additive"' 'explainer_normalization_class = None' 'explainer_link = "softmax"' 'explainer_head_num_attention_blocks = 1' 'explainer_head_include_cls = True' 'explainer_head_num_mlp_layers = 3' 'explainer_norm = True' 'explainer_freeze_backbone = "none"' 'explainer_head_mlp_layer_ratio = 4' 'explainer_activation="tanh"' \
'explainer_backbone_type = "vit_base_patch16_224"' 'explainer_download_weight = False' 'explainer_load_path = "results/wandb_transformer_interpretability_project/1ytqhm00/checkpoints/epoch=22-step=3380.ckpt"' \
'unfreeze_after=None' 'unfreeze_after_gradual = False' \
training_hyperparameters_transformer 'per_gpu_batch_size = 16' 'checkpoint_metric = "loss"' 'learning_rate = 1e-4' 'max_epochs = 100' 'precision = 32'
```

```bash
python main.py with 'stage = "explainer"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "ImageNette_vit_base_patch16_224_explainer_lr1e-4_additive_block1_clstrue_mlp3_ratio4_freezefalse_normtrue_softmax_acttanh_zeroinput_classifiermasked"' \
env_chanwkim 'gpus_surrogate=[7]' 'gpus_explainer=[0]' \
dataset_ImageNette \
'surrogate_mask_location = "zero-input"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/wandb_transformer_interpretability_project/1ytqhm00/checkpoints/epoch=22-step=3380.ckpt"' \
'explainer_num_mask_samples = 32' 'explainer_num_mask_samples_epoch = 0' 'explainer_paired_mask_samples = True' \
'explainer_normalization = "additive"' 'explainer_normalization_class = None' 'explainer_link = "softmax"' 'explainer_head_num_attention_blocks = 1' 'explainer_head_include_cls = True' 'explainer_head_num_mlp_layers = 3' 'explainer_norm = True' 'explainer_freeze_backbone = "none"' 'explainer_head_mlp_layer_ratio = 4' 'explainer_activation="tanh"' \
'explainer_backbone_type = "vit_base_patch16_224"' 'explainer_download_weight = True' \
'unfreeze_after=None' 'unfreeze_after_gradual = False' \
training_hyperparameters_transformer 'per_gpu_batch_size = 32' 'checkpoint_metric = "loss"' 'learning_rate = 1e-4' 'max_epochs = 100'
```

## Initialization comparison

```bash
python main.py with 'stage = "explainer"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "ImageNette_vit_base_patch16_224_explainer_lr1e-4_additive_block1_clstrue_mlp3_ratio4_freezefalse_normtrue_softmax_acttanh_initrandom"' \
env_chanwkim 'gpus_surrogate=[5]' 'gpus_explainer=[6]' \
dataset_ImageNette \
'surrogate_mask_location = "pre-softmax"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/wandb_transformer_interpretability_project/3i6zzjnp/checkpoints/epoch=38-step=5732.ckpt"' \
'explainer_num_mask_samples = 32' 'explainer_num_mask_samples_epoch = 0' 'explainer_paired_mask_samples = True' \
'explainer_normalization = "additive"' 'explainer_normalization_class = None' 'explainer_link = "softmax"' 'explainer_head_num_attention_blocks = 1' 'explainer_head_include_cls = True' 'explainer_head_num_mlp_layers = 3' 'explainer_norm = True' 'explainer_freeze_backbone = "none"' 'explainer_head_mlp_layer_ratio = 4' 'explainer_activation="tanh"' \
'explainer_backbone_type = "vit_base_patch16_224"' 'explainer_download_weight = False' 'explainer_load_path = None' \
'unfreeze_after=None' 'unfreeze_after_gradual = False' \
training_hyperparameters_transformer 'per_gpu_batch_size = 32' 'checkpoint_metric = "loss"' 'learning_rate = 1e-4' 'max_epochs = 100'
```

```bash
python main.py with 'stage = "explainer"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "ImageNette_vit_base_patch16_224_explainer_lr1e-4_additive_block1_clstrue_mlp3_ratio4_freezefalse_normtrue_softmax_acttanh_unet"' \
env_chanwkim 'gpus_surrogate=[6]' 'gpus_explainer=[7]' \
dataset_ImageNette \
'surrogate_mask_location = "pre-softmax"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/wandb_transformer_interpretability_project/3i6zzjnp/checkpoints/epoch=38-step=5732.ckpt"' \
'explainer_num_mask_samples = 32' 'explainer_num_mask_samples_epoch = 0' 'explainer_paired_mask_samples = True' \
'explainer_normalization = "additive"' 'explainer_normalization_class = None' 'explainer_link = "softmax"' 'explainer_head_num_attention_blocks = 1' 'explainer_head_include_cls = True' 'explainer_head_num_mlp_layers = 3' 'explainer_norm = True' 'explainer_freeze_backbone = "none"' 'explainer_head_mlp_layer_ratio = 4' 'explainer_activation="tanh"' \
'explainer_backbone_type = "unet"' 'explainer_download_weight = False' 'explainer_load_path = None' \
'unfreeze_after=None' 'unfreeze_after_gradual = False' \
training_hyperparameters_transformer 'per_gpu_batch_size = 32' 'checkpoint_metric = "loss"' 'learning_rate = 1e-4' 'max_epochs = 100'
```