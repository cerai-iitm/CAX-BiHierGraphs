import os
from dotenv import load_dotenv
import torch
import torch.multiprocessing as mp
import torch_geometric.transforms as T

from torch_geometric import seed_everything
from Explainer_utils import (
    hetero_src_tgt_khop_in_subgraph,
    k_shortest_paths_with_max_length,
)
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pickle
from tqdm import tqdm

from med_exp.xAI.get_human_readable_explanations import get_LLM_base_explanation, get_LLM_explanations_all, summarize_all_explanations
from med_exp.xAI.med_graph_visualizer import visualize
from med_exp.graph.med_graph_utils import getNodeText
import random, math

# Set seed before anything else
seed_everything(4321)


def getFeat(graph, mapping):
    temp = {}
    for node_type in graph.node_types:
        for i, node in enumerate(graph[node_type].x):
            temp[(node_type, i)] = getNodeText(mapping[node_type][i], node_type)
    return temp


load_dotenv()
hf_token = os.getenv("HF_TOKEN")

def initialize_llm():
    """Initialize the LLM across multiple GPUs"""
    # llm = HuggingFacePipeline.from_model_id(
    #     # model_id="YBXL/Med-LLaMA3-8B",
    #     # model_id = "google/gemma-7b",
    #     model_id = "BioMistral/BioMistral-7B",
    #     # model_id = "meta-llama/Meta-Llama-3-8B",
    #     # model_id = "ContactDoctor/Bio-Medical-Llama-3-8B",
    #     task="text-generation",
    #     pipeline_kwargs={
    #         "max_new_tokens": int(1e5),
    #         "top_k": 50,
    #         "temperature": 0.7,
    #         "do_sample": True,
    #         "repetition_penalty": 1.2,
    #     },
    #     device=0,
    #     device_map="auto"
    # )

    # tokenizer = AutoTokenizer.from_pretrained("YBXL/Med-LLaMA3-8B")

    # # Load the model with specific generation configurations
    # model = AutoModelForCausalLM.from_pretrained(
    #     "YBXL/Med-LLaMA3-8B",
    #     torch_dtype=torch.float16,  # Use float16 for memory efficiency
    #     device_map="auto"  # Automatically distribute across available GPUs
    # )

    # Create a text generation pipeline
    # llm = AutoModelForCausalLM(
    #     model=model,
    #     tokenizer=tokenizer,
    #     max_new_tokens=100000,
    #     do_sample=True,
    #     top_k=50,
    #     temperature=0.4,
    #     repetition_penalty=1.2
    # )

    model_id = "ContactDoctor/Bio-Medical-Llama-3-8B"

    hf_token = "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"
    # llm = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id,
        use_fast=True,
        token=hf_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", token=hf_token
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model


# Initialize LLM globally so all processes can use it
tokenizer, model = initialize_llm()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Device: {device}")


def main():
    """Main function to distribute tasks across GPUs"""
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    noDupEdges = False
    # Load graph and edge masks
    if noDupEdges:
        graph = pickle.load(
            open(
                "med_exp/graph/results/explanations/hetero_gnn_explainer/NoDupEdges/test_graph.pkl",
                "rb",
            )
        ).to(device)
    else:
        graph = pickle.load(
            open(
                "med_exp/graph/results/explanations/hetero_gnn_explainer/test_graph.pkl",
                "rb",
            )
        ).to(device)

    filenum = 190

    if noDupEdges:
        pickle_filename = f"med_exp/graph/results/explanations/hetero_gnn_explainer/NoDupEdges/gnnexp_gnn_hier_med_graph_3GATConv_pred_edge_to_comp_g_edge_mask_{filenum}.pkl"
    else:
        pickle_filename = f"med_exp/graph/results/explanations/hetero_gnn_explainer/gnnexp_gnn_hier_med_graph_3GATConv_pred_edge_to_comp_g_edge_mask_{filenum}.pkl"

    baseline_scores_array, hier_scores_array = [], []

    num_neighbors = 3
    variation = "search"

    valid_paths = 0
    examples = 0

    edge_masks = pickle.load(open(pickle_filename, "rb"))
    # for itr in range(0, 3):
    #     edge_masks = pickle.load(open(pickle_filename + f"_part{itr + 1}.pkl", "rb"))
    #     subset_links_to_explain = [
    #         link
    #         for link in links_to_explain
    #         if itr * items_per_part
    #         <= link
    #         < min((itr + 1) * items_per_part, total_items)
    #     ]
    #

    os.makedirs(
        "med_exp/graph/results/explanations/hetero_gnn_explainer", exist_ok=True
    )

    # links_to_explain = random.sample(range(0, len(edge_masks)), 50)
    links_to_explain = range(0, len(edge_masks))
    # print(f"Links to explain: {links_to_explain}")
    # links_to_explain = [86]

    link_idx = 0
    for nodes, mask in tqdm(edge_masks.items()):
        link_idx += 1
        print(link_idx)
        if link_idx not in links_to_explain:
            continue

        if noDupEdges:
            results_folder = (
                "med_exp/graph/results/explanations/hetero_gnn_explainer/noDupEdges/"
            )
        else:
            results_folder = "med_exp/graph/results/explanations/hetero_gnn_explainer/singleExp/"

        results_folder = f"{results_folder}/{link_idx}"
        os.makedirs(results_folder, exist_ok=True)

        # Process mask
        mask = {etype: mask[etype].detach().sigmoid().to(device) for etype in mask}

        src_ntype, src_nid = nodes[0][0], nodes[0][1].item()
        tgt_ntype, tgt_nid = nodes[1][0], nodes[1][1].item()

        # print(f"src_ntype: {src_ntype}, src_nid: {src_nid}, tgt_ntype: {tgt_ntype}, tgt_nid: {tgt_nid}")

        # Extract node text
        note_txt = getNodeText(src_nid, src_ntype)
        icd_txt = getNodeText(tgt_nid, tgt_ntype)

        # put note and icd text in results folder
        with open(f'{results_folder}/note.txt', 'w') as f:
            f.write(note_txt)
        with open(f'{results_folder}/icd.txt', 'w') as f:
            f.write(icd_txt)

        # Generate baseline explanation if it doesn't exist
        if not os.path.exists(f'{results_folder}/baseline{link_idx}.txt'):
            llm_response, baseline_scores = get_LLM_base_explanation(tokenizer, model, note_txt, icd_txt)
            with open(f'{results_folder}/baseline{link_idx}.txt', 'w') as f:
                f.write(llm_response)
        # print(baseline_scores)

        mapping = None
        # Generate explainer-based explanation if it doesn't exist
        if not os.path.exists(f"{results_folder}/gnnexp{link_idx}.txt"):
            # (comp_g_src_nid, comp_g_tgt_nid, comp_g_k_hop, mapping) = hetero_src_tgt_khop_in_subgraph(
            #     src_ntype, src_nid, tgt_ntype, tgt_nid, graph, 3, device=device
            # )

            # get the k-hop subgraph
            (comp_g_src_nid, comp_g_tgt_nid, comp_g_k_hop, mapping) = (
                hetero_src_tgt_khop_in_subgraph(
                    src_ntype, src_nid, tgt_ntype, tgt_nid, graph, 3, device=device
                )
            )

            # Checking if mask and graph dimensions are matching after KHop(remove later)
            # for etype in mask:
            #     print(etype, end=" ")
            #     if comp_g_k_hop[etype].edge_index.shape[1] != mask[etype].shape[0]:
            #         print("Not matching ", end=" ")
            #         print(
            #             graph[etype].edge_index.shape[1],
            #             comp_g_k_hop[etype].edge_index.shape[1],
            #             mask[etype].shape[0],
            #             end="",
            #         )
            #     else:
            #         print("MATCHING")
            #     print()

            # print(f"comp_g_src_nid: {comp_g_src_nid}, comp_g_tgt_nid: {comp_g_tgt_nid}")

            # get the subgraph node to original graph node mapping
            for ntype in mapping:
                rev_map = {sub: orig for orig, sub in mapping[ntype].items()}
                mapping[ntype] = rev_map

            if variation == "apply":
                mask = {
                    etype: (mask[etype].sigmoid() > 0.385).to(device) for etype in mask
                }

                # for etype in mask:
                #     edge_index = comp_g_k_hop[etype].edge_index
                #     E = edge_index.size(1)
                #     m = (
                #         mask[etype].numel()
                #         if hasattr(mask[etype], "numel")
                #         else len(mask[etype])
                #     )
                #     print(
                #         f"etype={etype} | edge_index.shape={tuple(edge_index.shape)} | mask.shape={m}"
                #     )
                #     if m != E:
                #         # show first/last few mask values to help debugging
                #         print(" mask[:10]:", mask[etype][:10])
                #         print(" mask[-10:]:", mask[etype][-10:])
                #         print(
                #             " #True in mask:",
                #             mask[etype].to(torch.bool).sum().item(),
                #         )
                #         print("---")

                # Applying mask
                emasked_dict = {
                    etype: torch.stack(
                        (
                            torch.masked_select(
                                comp_g_k_hop[etype].edge_index[0], mask[etype]
                            ),
                            torch.masked_select(
                                comp_g_k_hop[etype].edge_index[1], mask[etype]
                            ),
                        )
                    )
                    for etype in mask
                }

                comp_g = comp_g_k_hop.clone()
                for edge_type in comp_g_k_hop.edge_types:
                    comp_g[edge_type].edge_index = emasked_dict[edge_type]

                explanation_paths = k_shortest_paths_with_max_length(
                    comp_g, "notes", comp_g_src_nid, "icds", comp_g_tgt_nid
                )

                key_errors = set()
                for pidx, path in enumerate(explanation_paths):
                    for eidx, edge in enumerate(path):
                        src_nid, tgt_nid, edge_type = edge
                        if edge_type == "_":
                            key_errors.add(pidx)
                            continue
                        try:
                            explanation_paths[pidx][eidx] = (
                                mapping[edge_type[0]][src_nid],
                                mapping[edge_type[2]][tgt_nid],
                                edge_type,
                            )
                        except KeyError:
                            key_errors.add(pidx)
                explanation_paths = [
                    path
                    for i, path in enumerate(explanation_paths)
                    if i not in key_errors
                ]

                examples += 1
                if len(explanation_paths) > 0:
                    valid_paths += 1

            elif variation == "search":
                (comp_g_src_nid, comp_g_tgt_nid, comp_g, beam_search_mapping) = (
                    hetero_src_tgt_khop_in_subgraph(
                        src_ntype,
                        comp_g_src_nid,
                        tgt_ntype,
                        comp_g_tgt_nid,
                        comp_g_k_hop,
                        3,
                        device=device,
                        edge_weights=mask,
                        num_neighbors=num_neighbors,
                    )
                )
                print(f"comp_g_src_nid: {comp_g_src_nid}, comp_g_tgt_nid: {comp_g_tgt_nid}")

                # get the beam search output subgraph to original graph node mapping
                for ntype in beam_search_mapping:
                    rev_map = {
                        bsm: mapping[ntype][sub]
                        for sub, bsm in beam_search_mapping[ntype].items()
                    }
                    mapping[ntype] = rev_map

                for ntype in beam_search_mapping:
                    rev_map = {
                        sub: orig for orig, sub in beam_search_mapping[ntype].items()
                    }
                    beam_search_mapping[ntype] = rev_map

                # get explanation paths
                explanation_paths = k_shortest_paths_with_max_length(
                    comp_g, "notes", comp_g_src_nid, "icds", comp_g_tgt_nid
                )
                print(f"Explanation paths: {explanation_paths}")

                key_errors = set()
                for pidx, path in enumerate(explanation_paths):
                    for eidx, edge in enumerate(path):
                        src_nid, tgt_nid, edge_type = edge
                        if edge_type == "_":
                            key_errors.add(pidx)
                            continue
                        try:
                            explanation_paths[pidx][eidx] = (
                                mapping[edge_type[0]][src_nid],
                                mapping[edge_type[2]][tgt_nid],
                                edge_type,
                            )
                        except KeyError:
                            key_errors.add(pidx)
                explanation_paths = [
                    path
                    for i, path in enumerate(explanation_paths)
                    if i not in key_errors
                ]
                examples += 1
                if len(explanation_paths) > 0:
                    valid_paths += 1
                # print("Filtered explanation paths: ", explanation_paths)

                all_human_readable_explanations, all_scores = get_LLM_explanations_all(tokenizer, model, explanation_paths)

                # print(all_human_readable_explanations)
                # all_valid_explanations = []
                # for idx, human_readable_explanation in enumerate(all_human_readable_explanations):
                #     if idx not in key_errors:
                #         all_valid_explanations.append(human_readable_explanation)
                # if len(all_valid_explanations) > 0:
                #     summarized_explanation, hier_scores = summarize_all_explanations(tokenizer, model, all_valid_explanations, all_scores)
                #     # print(hier_scores)
                #     with open(f'{results_folder}/gnnexp{link_idx}.txt', 'w') as f:
                #         f.write(summarized_explanation)

                # take only first explanation
                if len(all_human_readable_explanations) > 0:
                    with open(f'{results_folder}/gnnexp{link_idx}.txt', 'w') as f:
                        f.write(all_human_readable_explanations[0])

                # # baseline_scores_array.append(baseline_scores)
                # hier_scores_array.append(hier_scores)

                ################################################################################################################################################################################################
                # visualize the explanation subgraph
                ################################################################################################################################################################################################
                # Get weights of edges after beam search
                # for edge_type in comp_g.edge_types:
                #     comp_g[edge_type]["mask"] = []
                #     src_type, rel_type, dst_type = edge_type
                #     edge_index = comp_g[edge_type].edge_index
                #     edge_index_after_khop = comp_g_k_hop[edge_type].edge_index
                #     ori_edge_index = graph[edge_type].edge_index
                #     for i in range(edge_index.size(1)):
                #         src, dst = edge_index[:, i].tolist()
                #         print(src, dst)

                #         src_ori = mapping[src_type][src]
                #         dst_ori = mapping[dst_type][dst]
                #         print(src_ori, dst_ori, edge_type)
                #         mask_index = (ori_edge_index[0] == src_ori) & (
                #             ori_edge_index[1] == dst_ori
                #         )
                #         print(mask_index.nonzero(as_tuple=False))

                #         src = beam_search_mapping[src_type][src]
                #         dst = beam_search_mapping[dst_type][dst]
                #         print(src, dst)

                #         mask_index = (edge_index_after_khop[0] == src) & (
                #             edge_index_after_khop[1] == dst
                #         )
                #         print(mask_index.nonzero(as_tuple=False))
                #         print(len(edge_index_after_khop[0]))
                #         print(
                #             edge_index_after_khop[0][34377],
                #             edge_index_after_khop[1][34377],
                #         )
                #         print(
                #             edge_index_after_khop[0][34406],
                #             edge_index_after_khop[1][34406],
                #         )
                #         print(mask[edge_type][34377], mask[edge_type][34406])

                #         edge_idx = (
                #             mask_index.nonzero(as_tuple=False).squeeze().item()
                #         )
                #         comp_g[edge_type]["mask"].append(
                #             mask[edge_type][edge_idx].item()
                #         )

                # if not os.path.exists(
                #     f"med_exp/graph/results/explanations/hetero_gnn_explainer/gnnexp{link_idx}.html"
                # ):
                #     feat_nodes = getFeat(comp_g, mapping)
                #     visualize(
                #         comp_g,
                #         "notes",
                #         comp_g_src_nid,
                #         "icds",
                #         comp_g_tgt_nid,
                #         feat_nodes,
                #         f"med_exp/graph/results/explanations/hetero_gnn_explainer/gnnexp{link_idx}.html",
                #     )

    # delete
    del edge_masks

    # Save the scores
    # with open('med_exp/graph/results/explanations/hetero_gnn_explainer/baseline_scores.pkl', 'wb') as f:
    #     pickle.dump(baseline_scores_array, f)
    # with open(
    #     f"med_exp/graph/results/explanations/hetero_gnn_explainer/hier_scores_{num_neighbors}.pkl",
    #     "wb",
    # ) as f:
    #     pickle.dump(hier_scores_array, f)
    # print("Scores saved")
    #
    # print(f"Examples {examples} and Valid Paths {valid_paths}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
