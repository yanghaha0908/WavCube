from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import torch

@rank_zero_only
def print_model(model):
    print(model)

if __name__ == "__main__":
    cli = LightningCLI(run=False)
    
    model = cli.model
    if hasattr(model.feature_extractor, 'stage') and model.feature_extractor.stage == 2:
        ckpt_path = model.feature_extractor.stage1_ckpt_path
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        s1_state_dict = checkpoint["state_dict"]  
        msg = model.load_state_dict(s1_state_dict, strict=False)
        
        print("missing_keys:")
        print(*(msg.missing_keys), sep="\n")
        print("unexpected_keys:")
        print(*(msg.unexpected_keys), sep="\n")

    print_model(cli.model)
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
