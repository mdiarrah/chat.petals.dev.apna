import os
import torch
from petals.constants import PUBLIC_INITIAL_PEERS

from data_structures import ModelBackendConfig, ModelChatConfig, ModelConfig, ModelFrontendConfig

default_chat_config = ModelChatConfig(
    max_session_length=8192,
    sep_token="###",
    stop_token="###",
    extra_stop_sequences=["</s>"],
    generation_params=dict(do_sample=1, temperature=0.6, top_p=0.9),
)

MODEL_FAMILIES = {
    "Llama 2": [
        ModelConfig(
            ModelBackendConfig(repository="meta-llama/Llama-2-70b-chat-hf"),
            ModelFrontendConfig(
                name="Llama 2 (70B-Chat)",
                model_card="https://huggingface.co/meta-llama/Llama-2-70b-chat-hf",
                license="https://bit.ly/llama2-license",
            ),
            default_chat_config,
        ),
    ],  
}
DEFAULT_MODEL_NAME = "meta-llama/Llama-2-70b-chat-hf"
HF_ACCESS_TOKEN = "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"
INITIAL_PEERS = []
BOOTSTRAP_PEERS = os.environ['INITIAL_PEERS']
if BOOTSTRAP_PEERS != "":
    INITIAL_PEERS.append(BOOTSTRAP_PEERS)
#INITIAL_PEERS = PUBLIC_INITIAL_PEERS
# Set this to a list of multiaddrs to connect to a private swarm instead of the public one, for example:
# INITIAL_PEERS = ['/ip4/10.1.2.3/tcp/31234/p2p/QmcXhze98AcgGQDDYna23s4Jho96n8wkwLJv78vxtFNq44']

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from cpufeature import CPUFeature

    has_avx512 = CPUFeature["AVX512f"] and CPUFeature["OS_AVX512"]
except ImportError:
    has_avx512 = False

if DEVICE == "cuda":
    TORCH_DTYPE = "auto"
elif has_avx512:
    TORCH_DTYPE = torch.bfloat16
else:
    TORCH_DTYPE = torch.float32  # You can use bfloat16 in this case too, but it will be slow

STEP_TIMEOUT = 5 * 60
