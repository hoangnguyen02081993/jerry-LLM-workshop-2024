from lib.model_v1 import ScamSMSClassifierV1
from lib.model_v2 import ScamSMSClassifierV2
from lib.model import IModel

class ModelProvider():
    def __init__(self):
        pass

    def get_model(self, model_version, config) -> IModel:
        if model_version == 1:
            return ScamSMSClassifierV1(config["vocab_size"], config["emb_dim"], config["hidden_dim"], config["num_classes"])
        elif model_version == 2:
            return ScamSMSClassifierV2(config['model_name'], config["num_classes"], config["dropout"])