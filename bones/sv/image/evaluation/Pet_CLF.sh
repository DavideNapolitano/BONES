python main.py with 'stage = "classifier"' \
'wandb_project_name = "wandb_transformer_interpretability_project_BONES"' 'exp_name = "Pet_classifier_vit_tiny_patch16_224_1e-5_train_BONES"' \
env_chanwkim 'gpus_classifier=[0]' \
dataset_Pet \
'classifier_backbone_type = "vit_tiny_patch16_224"' 'classifier_download_weight = True' 'classifier_load_path = None' \
training_hyperparameters_transformer 'checkpoint_metric = "accuracy"' 'learning_rate = 1e-5'