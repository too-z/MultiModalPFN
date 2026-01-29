import torch
from mmpfn.models.mmpfn.model.transformer import MultiheadGatedMLP, CrossAttentionPooler, PerFeatureTransformer

def count_parameters(model, model_name):
    """Counts the number of trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in {model_name}: {total_params:,}")
    return total_params

# Default values from PerFeatureTransformer context
DEFAULT_EMSIZE = 128
n_hid = DEFAULT_EMSIZE * 4
n_inp = DEFAULT_EMSIZE
mgm_heads = [1, 2,4,8,16,32,64,128]
cap_heads = [16]
encoder_dropout = 0.1

for mgm_head in mgm_heads:
    for cap_head in cap_heads:

        print(f"--- Counting parameters for standalone modules mgm:{mgm_head}, cap:{cap_head} ---")

        # Instantiate and count parameters for MultiheadGatedMLP
        mgm = MultiheadGatedMLP(in_dim=n_hid, out_dim=n_inp, mgm_heads=mgm_head, dropout=encoder_dropout)
        n_mgm = count_parameters(mgm, "MultiheadGatedMLP")

        # Instantiate and count parameters for CrossAttentionPooler
        cap = CrossAttentionPooler(src_dim=n_inp, cap_heads=cap_head, dropout=encoder_dropout)
        n_cap = count_parameters(cap, "CrossAttentionPooler")
        print("total params:", n_mgm + n_cap)


        # print("--- Counting parameters within PerFeatureTransformer instance ---")

        # try:
        #     # Minimal instantiation of PerFeatureTransformer
        #     whole_model = PerFeatureTransformer(
        #         mgm_heads=mgm_head,
        #         cap_heads=cap_head,
        #         mixer_type="MGM+CAP"
        #     )

        #     if hasattr(whole_model, 'mgm'):
        #         count_parameters(whole_model.mgm, "MultiheadGatedMLP (in PerFeatureTransformer)")
        #     else:
        #         print("No 'mgm' module in PerFeatureTransformer instance.")

        #     if hasattr(whole_model, 'cap'):
        #         count_parameters(whole_model.cap, "CrossAttentionPooler (in PerFeatureTransformer)")
        #     else:
        #         print("No 'cap' module in PerFeatureTransformer instance.")

        # except Exception as e:
        #     print(f"\nCould not instantiate PerFeatureTransformer to count submodules.")
        #     print(f"This is likely due to complex dependencies for the full model.")
        #     print(f"The counts above for the standalone modules are correct.")
        #     print(f"Error: {e}")
        print("")
