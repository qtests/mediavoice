from transformers.utils import logging
logging.set_verbosity_debug()

import transformers, torch
print("transformers:", transformers.__version__)
print("torch:", torch.__version__)

from transformers import HubertConfig, HubertModel

print("Finito!")

