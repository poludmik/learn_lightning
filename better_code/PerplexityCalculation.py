import torch
from collections import OrderedDict
from Model import LightningGemma2Module
from litgpt import Config
import warnings
from torchmetrics.text import Perplexity
from torch.utils.data import DataLoader
import pickle


class PPXDataLoader(DataLoader):
    def __init__(self, file_path, **kwargs):
        super().__init__(file_path, **kwargs)
        self.file_path = file_path
        # must be pickle file
        assert self.file_path.endswith(".pkl")
        self.data = pickle.load(open(self.file_path, "rb"))

    def __iter__(self):
        for i in range(len(self.data)):
            yield self.data[i]

    def __len__(self):
        return len(self.data)


dataloader = PPXDataLoader("/mnt/proj2/open-29-45/poludmik/czech-llm/dataset-playground/adam_test_corpus/native_czech_ppx.pkl")

# Load the checkpoint
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
    checkpoint = torch.load("gemma-2-2b-checkpoints/cp-epoch=0-step=35000.ckpt.consolidated")
    model_config = Config.from_name("gemma-2-2b")
    hp_config = checkpoint["hyper_parameters"]["hp_config"]
    model = LightningGemma2Module(model_config=model_config, hp_config=hp_config)
    model.configure_model()

# Remove 'module.' prefix from the state_dict keys
state_dict = checkpoint["state_dict"]
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace("module.", "")  # Strip the 'module.' prefix
    new_state_dict[new_key] = v

model.module.load_state_dict(new_state_dict)
model.module.eval()

raw_torch_model = model.module
model = None

raw_torch_model.to("cuda")

all_ppx = []

for batch in dataloader:
    torch.cuda.empty_cache()

    input_ids = batch['input_ids'].to("cuda").unsqueeze(0)
    labels = batch['labels'].to("cuda").unsqueeze(0)
    print("Length of input_ids:", input_ids.shape[1])

    with torch.no_grad():
        logits = raw_torch_model(input_ids)  # No need to use .view(-1) here
        perplexity = Perplexity().to("cuda")
        ppx = perplexity(logits, labels)
        print(ppx)
        all_ppx.append(ppx.item())
    
    del logits, input_ids, labels

print("Average perplexity:", sum(all_ppx) / len(all_ppx))
