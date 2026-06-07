### Apply the edge masks to the original graph to get the subgraph explanation
import gc
import os
import sys
import torch
from torch_geometric import seed_everything

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
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
from datasets import load_dataset
from Graph_utils import getNodeText
from get_human_readable_graph_explanations import (
    get_LLM_explanations_all,
    get_LLM_base_explanation,
    get_LLM_silver_explanation,
    summarize_all_explanations,
)
from Graph_utils import *
from tqdm import tqdm

seed_everything(4321)

# All graph operations run on CPU.
GRAPH_DEVICE = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

category = "tp"

EXPL_ROOT = os.path.join(
    PROJECT_ROOT, "legal_exp", "results", "explanations", "hetero_pg_explainer"
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

# Load graph and PGExplainer edge masks.
edge_masks = pickle.load(
    open(
        os.path.join(
            CATEGORY_DIR,
            # TODO: replace with the actual PGExplainer edge mask filename.
            "pgexp_gnn_hier_graph_3GATConv_pred_edge_to_comp_g_edge_mask.pkl",
        ),
        "rb",
    )
)
graph = pickle.load(open(os.path.join(CATEGORY_DIR, "graph.pkl"), "rb"))
graph = graph.to(GRAPH_DEVICE)

dataset = load_dataset("ecthr_cases", "violation-prediction")
dataset = dataset["train"].to_pandas()


def getFeat(graph, mapping):
    temp = {}
    for node_type in graph.node_types:
        for i, node in enumerate(graph[node_type].x):
            if node_type == "facts":
                temp[(node_type, i)] = getFact(mapping[node_type][i])
            elif node_type == "terms":
                temp[(node_type, i)] = getTerm(mapping[node_type][i])
            elif node_type == "articles":
                temp[(node_type, i)] = getArticle(mapping[node_type][i])
            elif node_type == "cases":
                temp[(node_type, i)] = f"Case_{i}\n"
                facts = df_cases.loc[mapping[node_type][i], "facts"]
                for fact in facts:
                    temp[(node_type, i)] += fact + "\n"
    return temp


def getSelectedFacts(comp_g, comp_g_src_nid, mapping):
    req_indices = (
        (comp_g[("facts", "part_of", "cases")].edge_index[1] == comp_g_src_nid)
        .nonzero()
        .reshape(-1)
    )
    facts_comp_g = torch.index_select(
        comp_g[("facts", "part_of", "cases")].edge_index, 1, req_indices
    )[0]
    facts_graph = []
    for i in range(facts_comp_g.shape[0]):
        temp = mapping["facts"][facts_comp_g[i].item()]
        facts_graph.append(fact_case_offset[temp][0])
    return facts_graph


def clear_gpu_cache():
    """
    Centralised GPU cache flush.
    Called after every LLM inference block so the PyTorch allocator releases
    cached-but-freed memory before the next forward pass.
    """
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

model_id = "Equall/Saul-7B-Instruct-v1"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Convert edge masks to CPU tensors.
edge_masks_cpu = {
    (
        (nodes[0][0], nodes[0][1]),   # (src_ntype, src_nid_int)
        (nodes[1][0], nodes[1][1]),   # (tgt_ntype, tgt_nid_int)
    ): {
        etype: mask_dict[etype].detach().sigmoid().cpu()
        for etype in mask_dict
    }
    for nodes, mask_dict in edge_masks.items()
}

del edge_masks
clear_gpu_cache()

links_to_skip = [6, 8, 26, 28, 40, 75, 69, 73, 82, 95]  # skip because they take up too much memory
links_done = []

if __name__ == "__main__":
    num_neighbors = 3
    variation = "search"

    counter = -1
    valid_paths = 0
    examples = 0

    COMPUTE_SCORES = True

    for nodes, mask in tqdm(edge_masks_cpu.items()):
        counter += 1
        if counter > 15:  # just collect 100 for now
            break
        if counter in links_to_skip or counter in links_done:
            continue

        # Nodes are now plain (ntype, int) tuples.
        src_ntype, src_nid = nodes[0]
        tgt_ntype, tgt_nid = nodes[1]

        case_num = fact_case_offset[src_nid][1]
        silver_rationales = dataset.loc[case_num, "silver_rationales"]
        all_facts = dataset.loc[case_num, "facts"]
        silver_rationale_facts = [all_facts[i] for i in silver_rationales]

        case_txt   = getNodeText(case_num, "cases")
        article_txt = getNodeText(tgt_nid, tgt_ntype)

        baseline_out = os.path.join(OUTPUT_DIR, f"baseline{counter}.txt")
        # if not os.path.exists(baseline_out):
        #     with torch.no_grad():
        #         llm_response, token_probs = get_LLM_base_explanation(
        #             tokenizer, model, case_txt, article_txt,
        #             compute_scores=COMPUTE_SCORES,
        #         )
        #     with open(baseline_out, "w") as f:
        #         f.write(llm_response)
        #     save_scores(f"baseline{counter}", token_probs)
        #     clear_gpu_cache()

        # silver_out = os.path.join(OUTPUT_DIR, f"silver{counter}.txt")
        # if not os.path.exists(silver_out):
        #     with torch.no_grad():
        #         llm_response, token_probs = get_LLM_silver_explanation(
        #             tokenizer, model, case_txt, article_txt, silver_rationale_facts,
        #             compute_scores=COMPUTE_SCORES,
        #         )
        #     with open(silver_out, "w") as f:
        #         f.write(llm_response)
        #     save_scores(f"silver{counter}", token_probs)
        #     clear_gpu_cache()

        pgexp_out = os.path.join(OUTPUT_DIR, f"pgexp{counter}.txt")
        if os.path.exists(pgexp_out):
            # Subgraph extraction fully on CPU.
            (comp_g_src_nid, comp_g_tgt_nid, comp_g_k_hop, mapping) = (
                hetero_src_tgt_khop_in_subgraph(
                    src_ntype, src_nid, tgt_ntype, tgt_nid, graph, 3,
                    device=GRAPH_DEVICE,
                )
            )

            # Invert mapping once: subgraph node id → original graph node id
            for ntype in mapping:
                mapping[ntype] = {sub: orig for orig, sub in mapping[ntype].items()}

            if variation == "apply":
                # Apply mask thresholds on CPU.
                thresholded_mask = {etype: (mask[etype] > 0.385) for etype in mask}
                emasked_dict = {
                    etype: torch.stack((
                        torch.masked_select(comp_g_k_hop[etype].edge_index[0], thresholded_mask[etype]),
                        torch.masked_select(comp_g_k_hop[etype].edge_index[1], thresholded_mask[etype]),
                    ))
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
                (comp_g_src_nid, comp_g_tgt_nid, comp_g, beam_search_mapping) = (
                    hetero_src_tgt_khop_in_subgraph(
                        src_ntype, comp_g_src_nid,
                        tgt_ntype, comp_g_tgt_nid,
                        comp_g_k_hop, 3,
                        device=GRAPH_DEVICE,
                        edge_weights=mask,
                        num_neighbors=num_neighbors,
                    )
                )

                # Compose the two mappings in one pass:
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

            # Free the large subgraph structures before LLM inference.
            del mapping, comp_g_k_hop

            examples += 1
            if explanation_paths:
                valid_paths += 1

            if explanation_paths:
                with torch.no_grad():
                    all_human_readable_explanations, path_scores = get_LLM_explanations_all(
                        tokenizer, model, explanation_paths,
                        compute_scores=COMPUTE_SCORES,
                    )
                # Save one score tensor per explanation path.
                for path_idx, path_token_probs in enumerate(path_scores):
                    save_scores(f"pgexp{counter}_path{path_idx}", path_token_probs)
                clear_gpu_cache()

                if all_human_readable_explanations:
                    with torch.no_grad():
                        summarized_explanation, summary_scores = summarize_all_explanations(
                            tokenizer, model,
                            all_human_readable_explanations,
                            scores=None,
                            compute_scores=COMPUTE_SCORES,
                        )
                    save_scores(f"pgexp{counter}", summary_scores)
                    clear_gpu_cache()

                    # with open(pgexp_out, "a") as f:
                    #     f.write(summarized_explanation)

        del mask
        clear_gpu_cache()

    print(f"Examples {examples} and Valid Paths {valid_paths}")