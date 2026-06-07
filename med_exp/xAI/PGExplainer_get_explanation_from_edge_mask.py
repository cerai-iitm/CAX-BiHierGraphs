# PGExplainer explanation script for medical experiments
# Generates natural language explanations from PGExplainer edge masks.

import gc
import os
import sys
import torch
from torch_geometric import seed_everything

# Project root setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from Explainer_utils import (
    hetero_src_tgt_khop_in_subgraph,
    k_shortest_paths_with_max_length,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
from med_exp.graph.med_graph_utils import getNodeText
from med_exp.xAI.get_human_readable_explanations import (
    get_LLM_explanations_all,
    summarize_all_explanations,
)
from tqdm import tqdm

seed_everything(4321)

# ── Constants ────────────────────────────────────────────────────────────────
GRAPH_DEVICE = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
COMPUTE_SCORES = temperature  # BUG FIX: was inside __main__ block but referenced at module level implicitly

# Category of edges to explain
category = "tp"

# Directory layout
EXPL_ROOT = os.path.join(
    PROJECT_ROOT, "med_exp", "graph", "results", "explanations", "hetero_pg_explainer", "NoDupEdges"
)
CATEGORY_DIR = os.path.join(EXPL_ROOT, category)
OUTPUT_DIR = os.path.join(CATEGORY_DIR, "text_explanations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Token-probability scores are stored as individual .pkl files, one per
# generated output, under a dedicated sub-directory.  Files are only written
# when COMPUTE_SCORES=True; the directory is created unconditionally so the
# path is always valid.
SCORES_DIR = os.path.join(OUTPUT_DIR, "scores")
os.makedirs(SCORES_DIR, exist_ok=True)


def clear_gpu_cache():
    """Release GPU memory after LLM inference."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def save_scores(stem: str, token_probs) -> None:
    """Persist per-token log-probabilities to SCORES_DIR/<stem>.pkl.

    ``token_probs`` is the CPU tensor returned by ``_generate()`` when
    ``compute_scores=True``, or ``None`` when scoring is disabled.
    Nothing is written in the ``None`` case so that the ``COMPUTE_SCORES=False``
    fast-path produces no output files and no extra I/O.

    Args:
        stem:        Filename stem, e.g. ``"baseline42"`` or ``"pgexp7"``.
                     The ``.pkl`` extension is appended automatically.
        token_probs: 1-D CPU float tensor of per-token log-probs, or ``None``.
    """
    if token_probs is None:
        return
    out_path = os.path.join(SCORES_DIR, f"{stem}.pkl")
    with open(out_path, "wb") as fh:
        pickle.dump(token_probs, fh)


def initialize_llm():
    """Initialize the LLM on available device(s)."""
    model_id = "ContactDoctor/Bio-Medical-Llama-3-8B"
    hf_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        token=hf_token,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model


def load_data():
    """Load graph and edge masks; return CPU-converted mask dict."""
    edge_masks_raw = pickle.load(
        open(
            os.path.join(
        CATEGORY_DIR,
        f"pgexp_gnn_hier_med_graph_3GATConv_pred_edge_to_comp_g_edge_mask_{category}.pkl",
            ),
            "rb",
        )
    )
    graph = pickle.load(open(os.path.join(CATEGORY_DIR, "graph.pkl"), "rb"))
    graph = graph.to(GRAPH_DEVICE)

    edge_masks_cpu = {
        (nodes[0], nodes[1]): {
            etype: mask_dict[etype].detach().sigmoid().cpu()
            for etype in mask_dict
        }
        for nodes, mask_dict in edge_masks_raw.items()
    }

    del edge_masks_raw
    clear_gpu_cache()
    return graph, edge_masks_cpu


def main():
    tokenizer, model = initialize_llm()
    graph, edge_masks_cpu = load_data()

    num_neighbors = 3
    variation = "search"   # "apply" | "search"
    valid_paths = 0
    examples = 0

    links_to_skip: set = set()
    links_done: set = set()

    counter = -1
    for nodes, mask in tqdm(edge_masks_cpu.items()):
        counter += 1
        if counter > 15:
            break
        if counter in links_to_skip or counter in links_done:
            continue


        src_ntype, src_nid = nodes[0]
        tgt_ntype, tgt_nid = nodes[1]

        case_txt = getNodeText(src_nid, src_ntype)
        article_txt = getNodeText(tgt_nid, tgt_ntype)

        pgexp_out = os.path.join(OUTPUT_DIR, f"pgexp{counter}.txt")
        if os.path.exists(pgexp_out):
            # First k-hop subgraph extraction
            (
                comp_g_src_nid,
                comp_g_tgt_nid,
                comp_g_k_hop,
                mapping,
            ) = hetero_src_tgt_khop_in_subgraph(
                src_ntype, src_nid, tgt_ntype, tgt_nid, graph, 3,
                device=GRAPH_DEVICE,
            )

            # Invert mapping: subgraph id → original graph id
            for ntype in mapping:
                mapping[ntype] = {sub: orig for orig, sub in mapping[ntype].items()}

            explanation_paths = []  # BUG FIX: initialise before conditional branches
                                    # so the remapping block below always has a value

            if variation == "apply":
                thresholded_mask = {etype: (mask[etype] > 0.385) for etype in mask}
                emasked_dict = {
                    etype: torch.stack(
                        (
                            torch.masked_select(
                                comp_g_k_hop[etype].edge_index[0],
                                thresholded_mask[etype],
                            ),
                            torch.masked_select(
                                comp_g_k_hop[etype].edge_index[1],
                                thresholded_mask[etype],
                            ),
                        )
                    )
                    for etype in thresholded_mask
                }
                comp_g = comp_g_k_hop.clone()
                for edge_type in comp_g_k_hop.edge_types:
                    comp_g[edge_type].edge_index = emasked_dict[edge_type]
                del emasked_dict, thresholded_mask

                explanation_paths = k_shortest_paths_with_max_length(
                    comp_g, "cases", comp_g_src_nid, "articles", comp_g_tgt_nid
                )
                del comp_g

            elif variation == "search":
                (
                    comp_g_src_nid,
                    comp_g_tgt_nid,
                    comp_g,
                    beam_search_mapping,
                ) = hetero_src_tgt_khop_in_subgraph(
                    src_ntype,
                    comp_g_src_nid,
                    tgt_ntype,
                    comp_g_tgt_nid,
                    comp_g_k_hop,
                    3,
                    device=GRAPH_DEVICE,
                    edge_weights=mask,
                    num_neighbors=num_neighbors,
                )

                # Compose mappings: beam-search subgraph id → original graph id
                for ntype in beam_search_mapping:
                    mapping[ntype] = {
                        bsm: mapping[ntype][sub]
                        for sub, bsm in beam_search_mapping[ntype].items()
                    }
                del beam_search_mapping

                explanation_paths = k_shortest_paths_with_max_length(
                    comp_g, "cases", comp_g_src_nid, "articles", comp_g_tgt_nid
                )
                del comp_g

            # Remap path node ids back to original graph ids
            key_errors = set()
            for pidx, path in enumerate(explanation_paths):
                for eidx, edge in enumerate(path):
                    src, tgt, edge_type = edge
                    if edge_type == "_":
                        key_errors.add(pidx)
                        continue
                    try:
                        explanation_paths[pidx][eidx] = (
                            mapping[edge_type[0]][src],
                            mapping[edge_type[2]][tgt],
                            edge_type,
                        )
                    except KeyError:
                        key_errors.add(pidx)
            explanation_paths = [
                p for i, p in enumerate(explanation_paths) if i not in key_errors
            ]

            del mapping, comp_g_k_hop

            examples += 1
            if explanation_paths:
                valid_paths += 1

            if explanation_paths:
                with torch.no_grad():
                    all_human_readable_explanations, path_scores = get_LLM_explanations_all(
                        tokenizer, model, explanation_paths
                    )
                # Save one score tensor per explanation path.
                for path_idx, path_token_probs in enumerate(path_scores):
                    save_scores(f"pgexp{counter}_path{path_idx}", path_token_probs)
                clear_gpu_cache()

                if all_human_readable_explanations:
                    with torch.no_grad():
                        summarized_explanation, summary_scores = summarize_all_explanations(
                            tokenizer,
                            model,
                            all_human_readable_explanations,
                            all_scores=[None]
                        )
                    save_scores(f"pgexp{counter}", summary_scores)
                    clear_gpu_cache()
                    # with open(pgexp_out, "a") as f:
                    #     f.write(summarized_explanation)

        del mask
        clear_gpu_cache()

    print(f"Examples: {examples}, Valid Paths: {valid_paths}")


if __name__ == "__main__":
    main()