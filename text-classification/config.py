model_version = 2

def get_config(): 
    if model_version == 1:
        return {
            "model_version": model_version,
            "dataset_version": 2,
            "vocab_size": 50257,
            "emb_dim": 100,
            "hidden_dim": 200,
            "lr": 0.0001,
            "num_classes": 2,
            "epochs": 1
        }
    elif model_version == 2:
        return {
            "model_version": model_version,
            "model_name": "xlm-roberta-base",
            "dataset_version": 1,
            "lr": 2e-5,
            "num_classes": 2,
            "dropout": 0.3,
            "epochs": 3
        }