import os, sys
import torch
from dotenv import load_dotenv

# Project root setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from torch_geometric import seed_everything
from utils.Explainer_utils import (
    hetero_src_tgt_khop_in_subgraph,
    k_shortest_paths_with_max_length,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
from tqdm import tqdm

from med_exp.xAI.get_human_readable_explanations import (
    get_LLM_base_explanation,
    get_LLM_explanations_all,
    summarize_all_explanations,
)
from med_exp.graph.med_graph_utils import getNodeText

# Set seed before anything else
seed_everything(4321)

load_dotenv()

COMPUTE_SCORES = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAPH_DEVICE = device
print(f"Device: {device}")

def clear_gpu_cache():
    """BUG FIX: function was called but never defined."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_scores(stem: str, token_probs) -> None:
    """Persist per-token log-probabilities to SCORES_DIR/<stem>.pkl.

    ``token_probs`` is the CPU tensor returned by ``_generate()`` when
    ``compute_scores=True``, or ``None`` when scoring is disabled.
    Nothing is written in the ``None`` case so that the ``COMPUTE_SCORES=False``
    fast-path produces no output files and no extra I/O.

    Args:
        stem:        Filename stem, e.g. ``"baseline42"`` or ``"gnnexp7"``.
                     The ``.pkl`` extension is appended automatically.
        token_probs: 1-D CPU float tensor of per-token log-probs, or ``None``.
    """
    if token_probs is None:
        return
    out_path = os.path.join(SCORES_DIR, f"{stem}.pkl")
    with open(out_path, "wb") as fh:
        pickle.dump(token_probs, fh)


def getFeat(graph, mapping):
    temp = {}
    for node_type in graph.node_types:
        for i, node in enumerate(graph[node_type].x):
            temp[(node_type, i)] = getNodeText(mapping[node_type][i], node_type)
    return temp


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


category = "tp"
BASE_EXPL_ROOT = os.path.join(
    "med_exp", "graph", "results", "explanations", "hetero_gnn_explainer", "NoDupEdges"
)
CATEGORY_DIR = os.path.join(BASE_EXPL_ROOT, category)
OUTPUT_DIR = os.path.join(CATEGORY_DIR, "text_explanations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Token-probability scores are stored as individual .pkl files, one per
# generated output, under a dedicated sub-directory.  Files are only written
# when COMPUTE_SCORES=True; the directory is created unconditionally so the
# path is always valid.
SCORES_DIR = os.path.join(OUTPUT_DIR, "scores")
os.makedirs(SCORES_DIR, exist_ok=True)


def main():
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    tokenizer, model = initialize_llm()

    noDupEdges = True
    num_neighbors = 3
    variation = "search"

    # Load graph
    graph = pickle.load(open(os.path.join(CATEGORY_DIR, "graph.pkl"), "rb"))
    graph = graph.to(GRAPH_DEVICE)

    # Load edge masks
    if noDupEdges:
        pickle_filename = (
            f"med_exp/graph/results/explanations/hetero_gnn_explainer/NoDupEdges/{category}/"
            f"gnnexp_gnn_hier_med_graph_3GATConv_pred_edge_to_comp_g_edge_mask_{category}.pkl"
        )
    else:
        pickle_filename = (
            f"med_exp/graph/results/explanations/hetero_gnn_explainer/{category}/"
            f"gnnexp_gnn_hier_med_graph_3GATConv_pred_edge_to_comp_g_edge_mask_{category}.pkl"
        )

    edge_masks = pickle.load(open(pickle_filename, "rb"))

    links_to_skip = []  # skip links that consume too much memory
    links_done = []

    valid_paths = 0
    examples = 0

    link_idx = 0
    for nodes, mask in tqdm(edge_masks.items()):
        link_idx += 1
        print(f"Processing link {link_idx}")

        if link_idx > 15:  # only do 100 for now
            break
        if link_idx in links_to_skip or link_idx in links_done:
            continue

        if noDupEdges:
            results_folder = (
                "med_exp/graph/results/explanations/hetero_gnn_explainer/noDupEdges/"
            )
        else:
            results_folder = (
                "med_exp/graph/results/explanations/hetero_gnn_explainer/singleExp/"
            )

        results_folder = f"{results_folder}/{link_idx}"
        os.makedirs(results_folder, exist_ok=True)

        # Process mask
        mask = {
            etype: mask[etype].detach().sigmoid().to(device) for etype in mask
        }

        src_ntype, src_nid = nodes[0][0], nodes[0][1].item()
        tgt_ntype, tgt_nid = nodes[1][0], nodes[1][1].item()

        # Sanity checks
        assert 0 <= src_nid < graph[src_ntype].num_nodes, (
            f"Invalid source node ID {src_nid} for type {src_ntype}"
        )
        assert 0 <= tgt_nid < graph[tgt_ntype].num_nodes, (
            f"Invalid target node ID {tgt_nid} for type {tgt_ntype}"
        )

        note_txt = getNodeText(src_nid, src_ntype)
        icd_txt = getNodeText(tgt_nid, tgt_ntype)

        # Save note and ICD text
        with open(f"{results_folder}/note.txt", "w") as f:
            f.write(note_txt)
        with open(f"{results_folder}/icd.txt", "w") as f:
            f.write(icd_txt)

        baseline_out = os.path.join(OUTPUT_DIR, f"baseline{link_idx}.txt")
        if os.path.exists(baseline_out):
            with torch.no_grad():
                llm_response, token_probs = get_LLM_base_explanation(
                    tokenizer,
                    model,
                    note_txt,
                    icd_txt
                )
            # with open(baseline_out, "w") as f:
            #     f.write(llm_response)
            save_scores(f"baseline{link_idx}", token_probs)
            clear_gpu_cache()

        gnnexp_out = os.path.join(OUTPUT_DIR, f"gnnexp{link_idx}.txt")
        if os.path.exists(gnnexp_out):

            # First k-hop subgraph extraction
            (
                comp_g_src_nid,
                comp_g_tgt_nid,
                comp_g_k_hop,
                mapping,
            ) = hetero_src_tgt_khop_in_subgraph(
                src_ntype,
                src_nid,
                tgt_ntype,
                tgt_nid,
                graph,
                3,
                device=GRAPH_DEVICE,
            )

            # Invert mapping: subgraph node id → original graph node id
            for ntype in mapping:
                mapping[ntype] = {sub: orig for orig, sub in mapping[ntype].items()}

            # Second subgraph extraction guided by edge weights
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
                comp_g, "notes", comp_g_src_nid, "icds", comp_g_tgt_nid
            )
            del comp_g

            # Remap path node ids back to original graph ids
            key_errors = set()
            for pidx, path in enumerate(explanation_paths):
                for eidx, edge in enumerate(path):
                    path_src, path_tgt, edge_type = edge
                    if edge_type == "_":
                        key_errors.add(pidx)
                        continue
                    try:
                        explanation_paths[pidx][eidx] = (
                            mapping[edge_type[0]][path_src],
                            mapping[edge_type[2]][path_tgt],
                            edge_type,
                        )
                    except KeyError:
                        key_errors.add(pidx)
            explanation_paths = [
                path for i, path in enumerate(explanation_paths) if i not in key_errors
            ]

            del mapping, comp_g_k_hop

            examples += 1
            if explanation_paths:
                valid_paths += 1

            if explanation_paths:
                with torch.no_grad():
                    all_human_readable_explanations, path_scores = get_LLM_explanations_all(
                        tokenizer,
                        model,
                        explanation_paths
                    )
                # Save one score tensor per explanation path.
                for path_idx, path_token_probs in enumerate(path_scores):
                    save_scores(f"gnnexp{link_idx}_path{path_idx}", path_token_probs)
                clear_gpu_cache()

                if all_human_readable_explanations:
                    with torch.no_grad():
                        summarized_explanation, summary_scores = summarize_all_explanations(
                            tokenizer,
                            model,
                            all_human_readable_explanations,
                            all_scores=[None]
                        )
                    save_scores(f"gnnexp{link_idx}", summary_scores)
                    clear_gpu_cache()

                    # with open(gnnexp_out, "a") as f:
                    #     f.write(summarized_explanation)

                    # Also write first explanation to per-link folder
                    # with open(f"{results_folder}/gnnexp{link_idx}.txt", "w") as f:
                    #     f.write(all_human_readable_explanations[0])

    del edge_masks
    print(f"Done. Examples processed: {examples}, with valid paths: {valid_paths}")


if __name__ == "__main__":
    main()