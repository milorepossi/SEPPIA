"""
Quick Modal test script to verify 2-instance ESMc-6B LoRA initialization,
forward/backward pass, VRAM consumption, and step latency on A100-80GB.
"""

import time
from pathlib import Path
import modal

app = modal.App("test-esmc-6b-lora")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = str(PROJECT_ROOT / "splits")
DATA_DIR = str(PROJECT_ROOT / "data")

hf_volume = modal.Volume.from_name("esmc-hf-cache", create_if_missing=True)

esmc_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch",
        "transformers",
        "esm==3.4.1.post1",
        "pandas",
        "numpy",
        "scikit-learn",
        "scipy",
        "tqdm",
    )
    .add_local_dir(SPLITS_DIR, remote_path="/root/splits")
    .add_local_dir(DATA_DIR, remote_path="/root/data")
)


@app.function(
    gpu="A100-80GB",
    image=esmc_image,
    volumes={"/root/.cache/huggingface": hf_volume},
    timeout=600,
)
def test_two_instance_lora_step():
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from esm.models.esmc import EsmcModel, EsmcTokenizer

    print("=" * 70)
    print("TESTING TWO-INSTANCE ESMC-6B LORA STEP")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Device: {device} ({gpu_name}) | Total VRAM: {total_mem:.2f} GB")

    class LoRALinear(nn.Module):
        def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
            super().__init__()
            self.base_linear = base_linear
            self.rank = rank
            self.scaling = alpha / rank

            for p in self.base_linear.parameters():
                p.requires_grad = False

            in_features = base_linear.in_features
            out_features = base_linear.out_features

            self.lora_A = nn.Parameter(
                torch.empty(rank, in_features, dtype=base_linear.weight.dtype, device=base_linear.weight.device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(out_features, rank, dtype=base_linear.weight.dtype, device=base_linear.weight.device)
            )
            self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            base_out = self.base_linear(x)
            lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B) * self.scaling
            return base_out + lora_out

    # 1. Load Tokenizer
    tokenizer = EsmcTokenizer.from_pretrained("biohub/ESMC-6B")

    # 2. Load Model Instance 1: Allele Model
    print("\n[1/4] Loading Allele ESMc-6B instance...")
    t0 = time.time()
    allele_model = EsmcModel.from_pretrained("biohub/ESMC-6B", dtype=torch.bfloat16, device="cuda")
    print(f"  Allele ESMc-6B loaded in {time.time() - t0:.2f}s | VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    # 3. Load Model Instance 2: Peptide Model
    print("\n[2/4] Loading Peptide ESMc-6B instance...")
    t0 = time.time()
    pep_model = EsmcModel.from_pretrained("biohub/ESMC-6B", dtype=torch.bfloat16, device="cuda")
    print(f"  Peptide ESMc-6B loaded in {time.time() - t0:.2f}s | VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    # Freeze base parameters
    for p in allele_model.parameters():
        p.requires_grad = False
    for p in pep_model.parameters():
        p.requires_grad = False

    # Attach LoRA to top 4 layers
    num_lora_layers = 4
    total_blocks = len(allele_model.transformer.blocks)
    lora_params = []

    print(f"\n[3/4] Attaching LoRA (r=8, alpha=16) to top {num_lora_layers} of {total_blocks} blocks...")
    for i in range(total_blocks - num_lora_layers, total_blocks):
        # Allele model
        b_allele = allele_model.transformer.blocks[i]
        b_allele.attn.out_proj = LoRALinear(b_allele.attn.out_proj, rank=8, alpha=16.0)
        lora_params.extend([b_allele.attn.out_proj.lora_A, b_allele.attn.out_proj.lora_B])

        # Peptide model
        b_pep = pep_model.transformer.blocks[i]
        b_pep.attn.out_proj = LoRALinear(b_pep.attn.out_proj, rank=8, alpha=16.0)
        lora_params.extend([b_pep.attn.out_proj.lora_A, b_pep.attn.out_proj.lora_B])

    total_lora_params = sum(p.numel() for p in lora_params)
    print(f"  Attached LoRA adapters! Total LoRA parameters: {total_lora_params:,} ({total_lora_params * 2 / 1024**2:.2f} MB in bf16)")

    # Downstream toy regressor
    d_model = allele_model.config.hidden_size
    head = nn.Sequential(
        nn.Linear(2 * d_model, 256),
        nn.GELU(),
        nn.Linear(256, 1)
    ).to(device=device, dtype=torch.bfloat16)

    trainable_params = lora_params + list(head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

    # 4. Dummy forward + backward step
    print("\n[4/4] Executing forward + backward test step...")
    dummy_peps = ["GILGFVFTL", "NLVPMVATV", "LLFGYPVYV", "GLCTLVAML"]
    dummy_hlas = [
        "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRA"
    ] * 4

    pep_toks = tokenizer(dummy_peps, return_tensors="pt", padding=False)["input_ids"].to(device)
    hla_toks = tokenizer(dummy_hlas, return_tensors="pt", padding=False)["input_ids"].to(device)
    targets = torch.tensor([1.2, 0.4, 2.8, 0.1], device=device, dtype=torch.bfloat16)

    t0 = time.time()
    pep_out = pep_model(pep_toks).last_hidden_state  # [4, 11, 2560]
    pep_emb = pep_out[:, 1:10, :].mean(dim=1)         # [4, 2560]

    hla_out = allele_model(hla_toks).last_hidden_state # [4, 184, 2560]
    hla_emb = hla_out[:, 1:183, :].mean(dim=1)         # [4, 2560]

    joint = torch.cat([pep_emb, hla_emb], dim=-1)
    preds = head(joint).squeeze(-1)
    loss = F.mse_loss(preds, targets)

    fwd_time = time.time() - t0
    print(f"  Forward pass: {fwd_time:.3f}s | Loss: {loss.item():.4f} | VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    t0 = time.time()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    bwd_time = time.time() - t0
    print(f"  Backward + Optimizer step: {bwd_time:.3f}s | Peak VRAM: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")

    # Verify gradients
    print("\nGradient Verification:")
    print(f"  Allele LoRA grad norm: {lora_params[0].grad.norm().item():.6f}")
    print(f"  Peptide LoRA grad norm: {lora_params[2].grad.norm().item():.6f}")
    print(f"  Base weights grad is None: {allele_model.transformer.blocks[0].attn.layernorm_qkv.weight.grad is None}")

    return {
        "status": "success",
        "forward_time_s": fwd_time,
        "backward_time_s": bwd_time,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3),
    }


@app.local_entrypoint()
def main():
    res = test_two_instance_lora_step.remote()
    print("\nResult from Modal:")
    print(res)
