# Prompting an LLM with some path explanations to generate human-readable text

from Graph_utils import getEdgeText
import torch


# Prompt Engineering based on the edge
def generate_prompt(explanation_path):
    prompt = ""
    for idx, edge in enumerate(explanation_path):
        src_text, tgt_text = getEdgeText(edge)
        if idx == 0:
            prompt += (
                "The following is the case against the plaintiff:\n" + src_text + "\n"
            )

        if edge[2] == ("cases", "violate", "articles"):
            prompt += "The following is an article violated by the previously mentioned case:\n"
        elif edge[2] == ("articles", "rev_violate", "cases"):
            prompt += "The following is a case that violated the previously mentioned article:\n"
        elif edge[2] == ("cases", "rev_part_of", "facts"):
            prompt += "The following is an important section of the previously mentioned case:\n"
        elif edge[2] == ("facts", "part_of", "cases"):
            prompt += "The following is a case containing the previously mentioned important section:\n"
        elif edge[2] == ("facts", "has", "terms"):
            prompt += "The following is a term in the previously mentioned important section:\n"
        elif edge[2] == ("terms", "rev_has", "facts"):
            prompt += "The following is an important section of a case that contains the previously mentioned term:\n"
        elif edge[2] == ("terms", "rev_has", "articles"):
            prompt += "The following is an article that contains the previously mentioned term:\n"
        elif edge[2] == ("articles", "has", "terms"):
            prompt += "The following is a term that is present in the previously mentioned article:\n"

        if idx == len(explanation_path) - 1:
            # prompt += "The following is the article you want to argue that the defendant violated:\n" + tgt_text + "\n|Using the cases mentioned as precedents, build an argument to prove that the defendant is guilty of violating the article just mentioned.\nOutput only the argument and nothing else."
            prompt += (
                "The following is the article you want to argue that the defendant violated:\n"
                + tgt_text
                + "Using the cases mentioned as precedents, build an argument to prove that the defendant is guilty of violating the article just mentioned. \nCiting relevant precedents using the cases I mentioned, give me only the argument in the following format: Introduction, Violation(s), Precedent(s), Conclusion."
            )
        else:
            # prompt += tgt_text + "\nOutput only your understandings and nothing else.\n|"
            prompt += tgt_text + "\n"

        # prompt += "\nGive the response in the following format:\n<Argument1> on the basis of inferred knowledge from <Entity1>\nAnd so forth. Each argument must have some justification that you derive strictly from the prompt\n"

    return prompt


# Prompt SaulLM for a single explanation
def get_LLM_explanation(tokenizer, model, explanation_path):
    prompt = generate_prompt(explanation_path)

    messages = [
        {
            "role": "user",
            "content": "You're a lawyer working on a case on Human Rights. You have to generate an argument to prove that the defendant has violated a European Court of Human Rights article using the information that you're given.\n"
            + prompt,
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(
        prompt, return_tensors="pt", padding=True, return_attention_mask=True
    ).to(model.device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    terminators = [t for t in terminators if t is not None]

    outputs = model.generate(
        input_ids["input_ids"],
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=input_ids["attention_mask"],
    )
    generated_sequence_ids = outputs.sequences[0]
    input_ids_len = len(
        input_ids["input_ids"][0]
    )  # Length of the original prompt's token IDs
    decoded_input = tokenizer.decode(
        generated_sequence_ids[:input_ids_len], skip_special_tokens=False
    )
    newly_generated_ids = generated_sequence_ids[input_ids_len:]

    scores = torch.nn.functional.softmax(
        torch.stack(outputs.scores, dim=0).squeeze(1), dim=-1
    )
    token_probs = scores[torch.arange(scores.size(0)), newly_generated_ids]
    token_probs = torch.log(token_probs + 1e-10)
    response = tokenizer.decode(newly_generated_ids, skip_special_tokens=True)
    return response, token_probs


def summarize_all_explanations(tokenizer, model, all_explanations, all_scores):
    if len(all_explanations) > 1:
        concatenated_explanations = " ".join(all_explanations)

        messages = [
            {
                "role": "user",
                "content": "Summarize all of the given information and give me ONLY the argument in the following format: Introduction, Prodecures, Similar Cases, Conclusion.\n"
                + concatenated_explanations,
            }
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(
            prompt, return_tensors="pt", padding=True, return_attention_mask=True
        ).to(model.device)
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        terminators = [t for t in terminators if t is not None]

        outputs = model.generate(
            input_ids["input_ids"],
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=input_ids["attention_mask"],
        )
        generated_sequence_ids = outputs.sequences[0]
        input_ids_len = len(
            input_ids["input_ids"][0]
        )  # Length of the original prompt's token IDs
        decoded_input = tokenizer.decode(
            generated_sequence_ids[:input_ids_len], skip_special_tokens=False
        )
        newly_generated_ids = generated_sequence_ids[input_ids_len:]

        scores = torch.nn.functional.softmax(
            torch.stack(outputs.scores, dim=0).squeeze(1), dim=-1
        )
        token_probs = scores[torch.arange(scores.size(0)), newly_generated_ids]
        token_probs = torch.log(token_probs + 1e-10)
        response = tokenizer.decode(newly_generated_ids, skip_special_tokens=True)
        return response, token_probs
    return all_explanations[0], all_scores[0]


# Collect all the explanations in a list: List[str]
def get_LLM_explanations_all(tokenizer, model, explanation_paths):
    all_explanations, scores = [], []
    for explanation_path in explanation_paths:
        human_readable_explanation, score = get_LLM_explanation(
            tokenizer, model, explanation_path
        )
        all_explanations.append(human_readable_explanation)
        scores.append(score)
    return all_explanations, scores


# Baseline explanation with only case and corresponding article
def get_LLM_base_explanation(tokenizer, model, case_txt, article_txt):
    raw_prompt = "The following is the case against the defandant:\n" + case_txt + "\n"
    raw_prompt += (
        "The following is the article you want to argue that the defendant violated:\n"
        + article_txt
        + "\n"
    )
    raw_prompt += "\nBuild an argument to prove that the defendant is guilty of violating the article mentioned. \nCiting relevant precedents, give me only the argument in the following format: Introduction, Violation(s), Precedent(s), Conclusion.\n"

    messages = [
        {
            "role": "user",
            "content": "You are now a lawyer working on a case on Human Rights. You have to generate arguments to prove that the defendant has violated a European Court of Human Rights article using the information that you're given.\n"
            + raw_prompt,
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(
        prompt, return_tensors="pt", padding=True, return_attention_mask=True
    ).to(model.device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    terminators = [t for t in terminators if t is not None]

    outputs = model.generate(
        input_ids["input_ids"],
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=input_ids["attention_mask"],
    )
    generated_sequence_ids = outputs.sequences[0]
    input_ids_len = len(
        input_ids["input_ids"][0]
    )  # Length of the original prompt's token IDs
    decoded_input = tokenizer.decode(
        generated_sequence_ids[:input_ids_len], skip_special_tokens=False
    )
    newly_generated_ids = generated_sequence_ids[input_ids_len:]

    scores = torch.nn.functional.softmax(
        torch.stack(outputs.scores, dim=0).squeeze(1), dim=-1
    )
    token_probs = scores[torch.arange(scores.size(0)), newly_generated_ids]
    token_probs = torch.log(token_probs + 1e-10)
    response = tokenizer.decode(newly_generated_ids, skip_special_tokens=True)
    return response, token_probs


# silver explanation with case, article and silver rationales
def get_LLM_silver_explanation(
    tokenizer, model, case_txt, article_txt, silver_rationales
):
    prompt = "The following is the case against the defandant:\n" + case_txt + "\n"
    prompt += (
        "The following is the article you want to argue that the defendant violated:\n"
        + article_txt
        + "\n"
    )

    prompt += "The following is a list of very important facts, each mentioned inside a <sr></sr> block, of the case that you must focus on while generating an argument:\n"
    for silver_rationale in silver_rationales:
        prompt += f"<sr>{silver_rationale}</sr>\n"

    prompt += "\nBuild an argument to prove that the defendant is guilty of violating the article mentioned. \nCiting relevant precedents, give me only the argument in the following format: Introduction, Violation(s), Precedent(s), Conclusion.\n"

    messages = [
        {
            "role": "user",
            "content": "You are now a lawyer working on a case on Human Rights. You have to generate arguments to prove that the defendant has violated a European Court of Human Rights article using the information that you're given.\n"
            + prompt,
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(
        prompt, return_tensors="pt", padding=True, return_attention_mask=True
    ).to(model.device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    terminators = [t for t in terminators if t is not None]

    outputs = model.generate(
        input_ids["input_ids"],
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=input_ids["attention_mask"],
    )
    generated_sequence_ids = outputs.sequences[0]
    input_ids_len = len(
        input_ids["input_ids"][0]
    )  # Length of the original prompt's token IDs
    decoded_input = tokenizer.decode(
        generated_sequence_ids[:input_ids_len], skip_special_tokens=False
    )
    newly_generated_ids = generated_sequence_ids[input_ids_len:]

    scores = torch.nn.functional.softmax(
        torch.stack(outputs.scores, dim=0).squeeze(1), dim=-1
    )
    token_probs = scores[torch.arange(scores.size(0)), newly_generated_ids]
    token_probs = torch.log(token_probs + 1e-10)
    response = tokenizer.decode(newly_generated_ids, skip_special_tokens=True)
    return response, token_probs
