# Prompting an LLM with some path explanations to generate human-readable text

import gc
import torch
from Graph_utils import getEdgeText


def _generate(tokenizer, model, messages, compute_scores=True):
    """
    Shared generation helper.
    """
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Encode on CPU, then move to model device in a single step.
    encoded = tokenizer(
        prompt_str,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    input_ids      = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    del encoded

    prompt_len = input_ids.shape[1]

    terminators = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None:
        terminators.append(eot_id)
    terminators = [t for t in terminators if t is not None]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            output_scores=False,          
            return_dict_in_generate=False,
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=attention_mask,
        )

    del input_ids, attention_mask

    # outputs is now a plain [1, total_seq_len] LongTensor
    generated_ids = outputs[0, prompt_len:]
    del outputs

    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Optional single-pass log-prob recovery.
    # When compute_scores=True we do one forward pass over the full
    # (prompt + response) sequence to get logits, then index to the
    # generated token positions. Peak VRAM is [seq_len, vocab_size] once,
    # immediately freed, rather than accumulating across generation steps.
    # When compute_scores=False this block is skipped entirely — saves
    # ~128 MB per call for a 32K vocab / 1024-token response.
    token_probs = None
    if compute_scores and generated_ids.numel() > 0:
        with torch.no_grad():
            full_ids  = tokenizer(
                prompt_str + response,
                return_tensors="pt",
                return_attention_mask=True,
            )
            full_input = full_ids["input_ids"].to(model.device)
            full_mask  = full_ids["attention_mask"].to(model.device)
            del full_ids

            logits = model(full_input, attention_mask=full_mask).logits
            del full_input, full_mask

            # logits[i] predicts token[i+1], so slice the generated portion
            gen_logits = logits[0, -(generated_ids.numel() + 1):-1, :]
            del logits

            log_probs = torch.nn.functional.log_softmax(gen_logits, dim=-1)
            del gen_logits

            token_probs = log_probs[
                torch.arange(generated_ids.numel(), device=log_probs.device),
                generated_ids.to(log_probs.device),
            ].cpu() # move to CPU so GPU can free this immediately
            del log_probs

    del generated_ids

    # flush GPU allocator cache inside the helper so every call
    # site benefits without relying on the outer loop to remember to flush.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response, token_probs


def generate_prompt(explanation_path):
    prompt = ""
    for idx, edge in enumerate(explanation_path):
        src_text, tgt_text = getEdgeText(edge)
        if idx == 0:
            prompt += "The following is the case against the plaintiff:\n" + src_text + "\n"

        edge_type = edge[2]
        if edge_type == ("cases", "violate", "articles"):
            prompt += "The following is an article violated by the previously mentioned case:\n"
        elif edge_type == ("articles", "rev_violate", "cases"):
            prompt += "The following is a case that violated the previously mentioned article:\n"
        elif edge_type == ("cases", "rev_part_of", "facts"):
            prompt += "The following is an important section of the previously mentioned case:\n"
        elif edge_type == ("facts", "part_of", "cases"):
            prompt += "The following is a case containing the previously mentioned important section:\n"
        elif edge_type == ("facts", "has", "terms"):
            prompt += "The following is a term in the previously mentioned important section:\n"
        elif edge_type == ("terms", "rev_has", "facts"):
            prompt += "The following is an important section of a case that contains the previously mentioned term:\n"
        elif edge_type == ("terms", "rev_has", "articles"):
            prompt += "The following is an article that contains the previously mentioned term:\n"
        elif edge_type == ("articles", "has", "terms"):
            prompt += "The following is a term that is present in the previously mentioned article:\n"

        if idx == len(explanation_path) - 1:
            prompt += (
                "The following is the article you want to argue that the defendant violated:\n"
                + tgt_text
                + "Using the cases mentioned as precedents, build an argument to prove that "
                "the defendant is guilty of violating the article just mentioned. \nCiting "
                "relevant precedents using the cases I mentioned, give me only the argument "
                "in the following format: Introduction, Violation(s), Precedent(s), Conclusion."
            )
        else:
            prompt += tgt_text + "\n"

    return prompt


def get_LLM_explanation(tokenizer, model, explanation_path, compute_scores=True):
    prompt = generate_prompt(explanation_path)
    messages = [
        {
            "role": "user",
            "content": (
                "You're a lawyer working on a case on Human Rights. You have to generate "
                "an argument to prove that the defendant has violated a European Court of "
                "Human Rights article using the information that you're given.\n" + prompt
            ),
        }
    ]
    return _generate(tokenizer, model, messages, compute_scores=compute_scores)


def summarize_all_explanations(
    tokenizer, model, all_explanations, scores=None, compute_scores=True
):
    """
    Summarise a list of per-path explanations into a single argument.
    """
    if len(all_explanations) == 1:
        return all_explanations[0], None

    concatenated = " ".join(all_explanations)
    messages = [
        {
            "role": "user",
            "content": (
                "Summarize all of the given information and give me ONLY the argument "
                "in the following format: Introduction, Procedures, Similar Cases, Conclusion.\n"
                + concatenated
            ),
        }
    ]
    return _generate(tokenizer, model, messages, compute_scores=compute_scores)


def get_LLM_explanations_all(tokenizer, model, explanation_paths, compute_scores=True):
    """
    Generate one explanation per path, processing them sequentially.
    """
    all_explanations, scores = [], []
    for explanation_path in explanation_paths:
        response, score = get_LLM_explanation(
            tokenizer, model, explanation_path, compute_scores=compute_scores
        )
        all_explanations.append(response)
        scores.append(score)   # None when compute_scores=False
    return all_explanations, scores


def get_LLM_base_explanation(
    tokenizer, model, case_txt, article_txt, compute_scores=True
):
    raw_prompt = (
        "The following is the case against the defendant:\n" + case_txt + "\n"
        "The following is the article you want to argue that the defendant violated:\n"
        + article_txt + "\n"
        "\nBuild an argument to prove that the defendant is guilty of violating the "
        "article mentioned. \nCiting relevant precedents, give me only the argument "
        "in the following format: Introduction, Violation(s), Precedent(s), Conclusion.\n"
    )
    messages = [
        {
            "role": "user",
            "content": (
                "You are now a lawyer working on a case on Human Rights. You have to "
                "generate arguments to prove that the defendant has violated a European "
                "Court of Human Rights article using the information that you're given.\n"
                + raw_prompt
            ),
        }
    ]
    return _generate(tokenizer, model, messages, compute_scores=compute_scores)


def get_LLM_silver_explanation(
    tokenizer, model, case_txt, article_txt, silver_rationales, compute_scores=True
):
    prompt = (
        "The following is the case against the defendant:\n" + case_txt + "\n"
        "The following is the article you want to argue that the defendant violated:\n"
        + article_txt + "\n"
        "The following is a list of very important facts, each mentioned inside a "
        "<sr></sr> block, of the case that you must focus on while generating an argument:\n"
    )
    for sr in silver_rationales:
        prompt += f"<sr>{sr}</sr>\n"
    prompt += (
        "\nBuild an argument to prove that the defendant is guilty of violating the "
        "article mentioned. \nCiting relevant precedents, give me only the argument "
        "in the following format: Introduction, Violation(s), Precedent(s), Conclusion.\n"
    )
    messages = [
        {
            "role": "user",
            "content": (
                "You are now a lawyer working on a case on Human Rights. You have to "
                "generate arguments to prove that the defendant has violated a European "
                "Court of Human Rights article using the information that you're given.\n"
                + prompt
            ),
        }
    ]
    return _generate(tokenizer, model, messages, compute_scores=compute_scores)