python main.py with 'stage = "explainer"' \
'wandb_project_name = "wandb_transformer_interpretability_project_BONES"' 'exp_name = "ImageNette_explainer_BONES"' \
env_chanwkim 'gpus_surrogate=[0]' 'gpus_explainer=[0]' \
dataset_ImageNette \
'surrogate_mask_location = "pre-softmax"' \
'surrogate_backbone_type = "vit_tiny_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/wandb_transformer_interpretability_project_BONES/ImageNette_surrogate_BONES/checkpoints/epoch=39-step=5879.ckpt"' \
'explainer_num_mask_samples = 32' 'explainer_num_mask_samples_epoch = 0' 'explainer_paired_mask_samples = True' \
'explainer_normalization = "additive"' 'explainer_normalization_class = None' 'explainer_link = "softmax"' 'explainer_head_num_attention_blocks = 1' 'explainer_head_include_cls = True' 'explainer_head_num_mlp_layers = 3' 'explainer_norm = True' 'explainer_freeze_backbone = "none"' 'explainer_head_mlp_layer_ratio = 4' 'explainer_activation="tanh"' \
'explainer_backbone_type = "vit_tiny_patch16_224"' 'explainer_download_weight = False' 'explainer_load_path = "results/wandb_transformer_interpretability_project_BONES/ImageNette_surrogate_BONES/checkpoints/epoch=39-step=5879.ckpt"' \
'unfreeze_after=None' 'unfreeze_after_gradual = False' \
training_hyperparameters_transformer 'per_gpu_batch_size = 32' 'checkpoint_metric = "loss"' 'learning_rate = 1e-4' 'max_epochs = 100'