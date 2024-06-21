python main.py with 'stage = "surrogate"' \
'wandb_project_name = "wandb_transformer_interpretability_project_BONES"' 'exp_name = "ImageNette_surrogate_BONES"' \
env_chanwkim 'gpus_classifier=[0]' 'gpus_surrogate=[0]' \
dataset_ImageNette \
'classifier_backbone_type = "vit_tiny_patch16_224"' 'classifier_download_weight = False' 'classifier_load_path = "results/wandb_transformer_interpretability_project_BONES/ImageNette_classifier_BONES/checkpoints/epoch=23-step=3527.ckpt"' \
'surrogate_mask_location = "pre-softmax"' \
'surrogate_backbone_type = "vit_tiny_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/wandb_transformer_interpretability_project_BONES/ImageNette_classifier_BONES/checkpoints/epoch=23-step=3527.ckpt"' \
training_hyperparameters_transformer 'checkpoint_metric = "loss"' 'learning_rate = 1e-5' 'max_epochs = 50'