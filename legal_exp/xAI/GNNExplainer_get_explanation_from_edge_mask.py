### Apply the edge masks to the original graph to get the subgraph explanation
import os
import torch
from torch_geometric import seed_everything
from Explainer_utils import (
    hetero_src_tgt_khop_in_subgraph,
    k_shortest_paths_with_max_length,
)
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pickle
from datasets import load_dataset
from graph_visualizer import visualize
from Graph_utils import getNodeText
from get_human_readable_graph_explanations import (
    get_LLM_explanations_all,
    get_LLM_base_explanation,
    get_LLM_silver_explanation,
    summarize_all_explanations,
)
from Graph_utils import *
from tqdm import tqdm
import random

seed_everything(4321)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

edge_masks = pickle.load(
    open(
        "./legal_exp/results/explanations/hetero_gnn_explainer/gnnexp_gnn_hier_graph_3GATConv_pred_edge_to_comp_g_edge_mask.pkl",
        "rb",
    )
)
graph = pickle.load(open("./legal_exp/results/explanations/hetero_gnn_explainer/graph.pkl", "rb"))
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


# llm = HuggingFacePipeline.from_model_id(
#     model_id="Equall/Saul-7B-Instruct-v1",
#     task="text-generation",
#     pipeline_kwargs={
#         "max_new_tokens": 1000,
#         "top_k": 50,
#         "temperature": 0.25,
#         "do_sample": True,
#         "repetition_penalty": 1.2,  # Added to discourage repetition
#     },
#     device=0
# )

# model_id = "Equall/Saul-7B-Instruct-v1"
# tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id, torch_dtype=torch.bfloat16, device_map="auto"
# )

# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token_id = tokenizer.eos_token_id

if __name__ == "__main__":
    with open("./datasets/case_fact_graph.pkl", "rb") as file:
        case_fact_graph = pickle.load(file)
    fact_case_offset = {}
    offset = 0
    prev_case = -1
    for i in range(len(case_fact_graph["case_fact_edges"])):
        if case_fact_graph["case_fact_edges"][i][1] != prev_case:
            offset = 0
        fact_case_offset[i] = [offset, case_fact_graph["case_fact_edges"][i][1]]
        prev_case = case_fact_graph["case_fact_edges"][i][1]
        offset += 1
    del case_fact_graph

    # links: 3, 7, 10, 13, 14, 16, 18, 21, 404, 407, 413, 414, 416, 1823, 2549, 2550
    # links_to_explain = [3, 7, 10, 13, 14, 16, 18, 21, 404, 407, 413, 414, 416, 1823, 2549, 2550]
    links_to_explain = [
        3,
        10,
        16,
        18,
        21,
        69,
        407,
        413,
        416,
        2549,
        1041,
        228,
        1669,
        724,
        1496,
        464,
        173,
        761,
        2402,
        217,
        1191,
        1682,
        2469,
        948,
        1778,
        1825,
        699,
        27,
        2178,
        1793,
        2411,
        863,
        1780,
        7,
        14,
        414,
        631,
        1823,
        2551,
    ]
    # removed but worked before:
    # removed: 260, 404, 466, 959, 1382, 1643, 1843, 1992, 2299, 2550, 2563, 13
    more_links = [261, 467, 960, 1383, 1644, 1844, 1993, 2300, 2564]
    links_to_explain.extend(more_links)

    # links_to_explain = [16, 217, 407, 413, 724, 1778, 1793, 1825, 1993, 2551]
    # links_to_explain = range(0, 1500) # all links

    # get a total of 50 links to explain
    # itrs = 1000 # stopping criterion
    # while len(links_to_explain) < 50 and itrs > 0:
    #     itrs -= 1
    #     rnd_idx = random.randint(0, 2597) # inclusive
    #     if rnd_idx not in links_to_explain:
    #         links_to_explain.append(rnd_idx)
    # print('List of links to explain calculated: ', links_to_explain)

    # two variations: apply edge mask or search for the top-k neighbors using the mask weights
    num_neighbors = 3
    variation = "search"  # options: search, apply

    counter = -1

    # store selected facts to calculate a retrieval metrics
    selected_facts = {}

    baseline_scores_array, silver_scores_array, hier_scores_array = [], [], []

    valid_paths = 0
    examples = 0

    for nodes, mask in tqdm(edge_masks.items()):
        counter += 1
        print(counter)
        # if counter not in links_to_explain:
        #     continue

        mask = {etype: mask[etype].detach().sigmoid().to(device) for etype in mask}

        src_ntype, src_nid = nodes[0][0], nodes[0][1].item()
        tgt_ntype, tgt_nid = nodes[1][0], nodes[1][1].item()

        # get the silver rationale facts
        case_num = fact_case_offset[src_nid][1]
        silver_rationales = dataset.loc[case_num, "silver_rationales"]
        silver_rationale_facts = dataset.loc[case_num, "facts"][silver_rationales]

        # case and article extraction
        case_txt = getNodeText(src_nid, src_ntype)
        article_txt = getNodeText(tgt_nid, tgt_ntype)

        # baseline explanation
        # if not os.path.exists(f'results/explanations/hetero_gnn_explainer/baseline{counter}.txt'):
        #     llm_response, baseline_scores = get_LLM_base_explanation(tokenizer, model, case_txt, article_txt)
        # with open(f'results/explanations/hetero_gnn_explainer/baseline{counter}.txt', 'w') as f:
        #     f.write(llm_response)

        # silver rationale explanation
        # if not os.path.exists(f'results/explanations/hetero_gnn_explainer/silver{counter}.txt'):
        #     llm_response, silver_scores = get_LLM_silver_explanation(tokenizer, model, case_txt, article_txt, silver_rationale_facts)
        # with open(f'results/explanations/hetero_gnn_explainer/silver{counter}.txt', 'w') as f:
        #     f.write(llm_response)

        # HierLegalGraph Explanation
        if (
            not os.path.exists(
                f"results/explanations/hetero_gnn_explainer/gnnexp{counter}.txt"
            )
            or True
        ):
            # get the k-hop subgraph
            (comp_g_src_nid, comp_g_tgt_nid, comp_g_k_hop, mapping) = (
                hetero_src_tgt_khop_in_subgraph(
                    src_ntype, src_nid, tgt_ntype, tgt_nid, graph, 3, device=device
                )
            )

            # get the subgraph node to original graph node mapping
            for ntype in mapping:
                rev_map = {sub: orig for orig, sub in mapping[ntype].items()}
                mapping[ntype] = rev_map

            if variation == "apply":
                mask = {
                    etype: (mask[etype].sigmoid() > 0.385).to(device) for etype in mask
                }

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
                    comp_g, "cases", comp_g_src_nid, "articles", comp_g_tgt_nid
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

                # get the selected facts
                # if src_nid in selected_facts:
                #     selected_facts[src_nid].update(set(getSelectedFacts(comp_g, comp_g_src_nid, mapping)))
                # else:
                #     selected_facts[src_nid] = set(getSelectedFacts(comp_g, comp_g_src_nid, mapping))

                # Get weights of edges after beam search
                # for edge_type in comp_g.edge_types:
                #     comp_g[edge_type]["mask"] = []
                #     src_type, rel_type, dst_type = edge_type
                #     edge_index = comp_g[edge_type].edge_index
                #     edge_index_after_khop = comp_g_k_hop[edge_type].edge_index

                #     for i in range(edge_index.size(1)):
                #         src, dst = edge_index[:, i].tolist()
                #         src = beam_search_mapping[src_type][src]
                #         dst = beam_search_mapping[dst_type][dst]

                #         mask_index = (edge_index_after_khop[0] == src) & (edge_index_after_khop[1] == dst)
                #         edge_idx = mask_index.nonzero(as_tuple=False).squeeze().item()
                #         comp_g[edge_type]["mask"].append(mask[edge_type][edge_idx].item())

                # get explanation paths
                explanation_paths = k_shortest_paths_with_max_length(
                    comp_g, "cases", comp_g_src_nid, "articles", comp_g_tgt_nid
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

                # all_human_readable_explanations, scores = get_LLM_explanations_all(tokenizer, model, explanation_paths)

                # all_valid_explanations = []
                # for idx, human_readable_explanation in enumerate(all_human_readable_explanations):
                #     if idx not in key_errors:
                #         all_valid_explanations.append(human_readable_explanation)
                # if len(all_valid_explanations) > 0:
                #     summarized_explanation, hier_scores = summarize_all_explanations(tokenizer, model, all_valid_explanations, scores)
                #     with open(f'results/explanations/hetero_gnn_explainer/gnnexp{counter}.txt', 'a') as f:
                #         f.write(summarized_explanation)

                # visualize the explanation subgraph
                # if not os.path.exists(f'results/explanations/hetero_gnn_explainer/gnnexp{counter}.html'):
                    # feat_nodes = getFeat(comp_g, mapping)
                    # visualize(comp_g,
                    #         'cases', comp_g_src_nid,
                    #         'articles', comp_g_tgt_nid,
                    #         feat_nodes, f'results/explanations/hetero_gnn_explainer/gnnexp{counter}.html'
                    #         )

                # baseline_scores_array.append(baseline_scores)
                # silver_scores_array.append(silver_scores)
                # hier_scores_array.append(hier_scores)

    # save scores
    # pickle.dump(baseline_scores_array, open('results/explanations/hetero_gnn_explainer/baseline_scores_gnnexp.pkl', 'wb'))
    # pickle.dump(silver_scores_array, open('results/explanations/hetero_gnn_explainer/silver_scores_gnnexp.pkl', 'wb'))
    # pickle.dump(hier_scores_array, open(f'results/explanations/hetero_gnn_explainer/hier_scores_gnnexp_{num_neighbors}.pkl', 'wb'))

    # pickle.dump(selected_facts, open('results/explanations/hetero_gnn_explainer/selected_facts_gnnexp.pkl', 'wb'))

    print(f"Examples {examples} and Valid Paths {valid_paths}")
