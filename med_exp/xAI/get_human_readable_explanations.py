# Prompting an LLM with some path explanations to generate human-readable text

from ..graph.med_graph_utils import getEdgeText
import torch

# Prompt Engineering based on the edge
def generate_prompt(explanation_path):
    prompt = ""
    for idx, edge in enumerate(explanation_path):
        src_text, tgt_text = getEdgeText(edge)
        if idx == 0:
            prompt += "The following is the medical case of the patient:\n" + src_text + "\n"
        if edge[2] == ('notes', 'links', 'icds'):
            prompt += "The following are the medical procedures that the previous case required:\n"
        elif edge[2] == ('icds', 'rev_links', 'notes'):
            prompt += "The following is a medical case that required the previously mentioned medical procedures:\n"
        elif edge[2] == ('icds', 'has', 'terms'):
            prompt += "The following is an important term in the previously mentioned medical procedures:\n"
        elif edge[2] == ('terms', 'rev_has', 'icds'):
            prompt += "The following are the medical procedures that contained the previously mentioned term:\n"
        elif edge[2] == ('notes', 'has', 'terms'):
            prompt += "The following is an important term in the previously mentioned note:\n"
        elif edge[2] == ('terms', 'rev_has', 'notes'):
            prompt += "The following is a medical case that contains the previously mentioned term:\n"
            
        prompt += tgt_text
            
        if idx == len(explanation_path)-1:
            # prompt += "The following is the article you want to argue that the defendant violated:\n" + tgt_text + "\n|Using the cases mentioned as precedents, build an argument to prove that the defendant is guilty of violating the article just mentioned.\nOutput only the argument and nothing else."
            prompt += "The following are the procedures you want to argue that the patient needs:\n" + tgt_text + "\nBuild an argument to prove that the patient needed the procedure mentioned earlier. \nCiting relevant parallels/precedents, give me ONLY the argument in the following format: Introduction, Procedures, Similar Cases, Conclusion\n"
        else:
            prompt += tgt_text + "\n"
            
    return prompt

# Prompt Medical LLM for a single explanation
def get_LLM_explanation(tokenizer, model, explanation_path):
    prompt = generate_prompt(explanation_path)
    
    messages = [
        {"role": "system", "content": "You are now a medical insurance agent working on a case on medical necessities. You have to generate arguments to prove that the medical procedures administered by the hospital are deemed a medical necessity using the information that you're given."},
        {"role": "user", "content": prompt}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(model.device)
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    terminators = [t for t in terminators if t is not None]
    
    outputs = model.generate(
        input_ids["input_ids"],
        max_new_tokens=4096,
        # eos_token_id=terminators,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=input_ids.get("attention_mask", None)
    )
    generated_sequence_ids = outputs.sequences[0]
    input_ids_len = len(input_ids["input_ids"][0]) # Length of the original prompt's token IDs
    decoded_input = tokenizer.decode(generated_sequence_ids[:input_ids_len], skip_special_tokens=False)
    newly_generated_ids = generated_sequence_ids[input_ids_len:]
    
    scores = torch.nn.functional.softmax(torch.stack(outputs.scores, dim=0).squeeze(1), dim=-1)
    token_probs = scores[torch.arange(scores.size(0)), newly_generated_ids]
    token_probs = torch.log(token_probs + 1e-10)
    response = tokenizer.decode(newly_generated_ids, skip_special_tokens=True)
    return response, token_probs

def summarize_all_explanations(tokenizer, model, all_explanations, all_scores):
    if len(all_explanations) > 1:
        concatenated_explanations = " ".join(all_explanations)
        
        messages = [
            {"role": "system", "content": "Summarize all of the given information and give me ONLY the argument in the following format: Introduction, Prodecures, Similar Cases, Conclusion."},
            {"role": "user", "content": concatenated_explanations}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True).to(model.device)
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
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
            attention_mask=input_ids["attention_mask"]
        )
        generated_sequence_ids = outputs.sequences[0]
        input_ids_len = len(input_ids["input_ids"][0]) # Length of the original prompt's token IDs
        decoded_input = tokenizer.decode(generated_sequence_ids[:input_ids_len], skip_special_tokens=False)
        newly_generated_ids = generated_sequence_ids[input_ids_len:]
        
        scores = torch.nn.functional.softmax(torch.stack(outputs.scores, dim=0).squeeze(1), dim=-1)
        token_probs = scores[torch.arange(scores.size(0)), newly_generated_ids]
        token_probs = torch.log(token_probs + 1e-10)
        response = tokenizer.decode(newly_generated_ids, skip_special_tokens=True)
        return response, token_probs
    return all_explanations[0], all_scores[0]

# Collect all the explanations in a list: List[str]
def get_LLM_explanations_all(tokenizer, model, explanation_paths):
    all_explanations, scores = [], []
    for explanation_path in explanation_paths:
        human_readable_explanation, score = get_LLM_explanation(tokenizer, model, explanation_path)
        all_explanations.append(human_readable_explanation)
        scores.append(score)
    return all_explanations, scores

# Baseline explanation with only note and corresponding icd
def get_LLM_base_explanation(tokenizer, model, note_txt, icd_txt):
    prompt = "The following is the medical case of the patient:\n" + note_txt + "\n"
    prompt += "The following are the medical procedures you want to argue that the patient needed:\n" + icd_txt + "\n"
    prompt += "\nBuild an argument to prove that the patient needed the procedure mentioned earlier. \nCiting relevant parallels, give me ONLY the argument in the following format: Introduction, Prodecures, Similar Cases, Conclusion\n"
    # response = llm.invoke(prompt)
    # response = llm(prompt)[0]['generated_text']
    # substring_index = response.find("[/INST]")
    # response = response[substring_index+7:] if substring_index != -1 else response
    
    messages = [
        {"role": "system", "content": "You are now a medical insurance agent working on a case on medical necessities. You have to generate arguments to prove that the medical procedures administered by the hospital are deemed a medical necessity using the information that you're given."},
        {"role": "user", "content": prompt}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True).to(model.device)
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    terminators = [t for t in terminators if t is not None]
    
    outputs = model.generate(
        input_ids["input_ids"],
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=input_ids["attention_mask"]
    )
    generated_sequence_ids = outputs.sequences[0]
    input_ids_len = len(input_ids["input_ids"][0]) # Length of the original prompt's token IDs
    decoded_input = tokenizer.decode(generated_sequence_ids[:input_ids_len], skip_special_tokens=False)
    newly_generated_ids = generated_sequence_ids[input_ids_len:]
    
    scores = torch.nn.functional.softmax(torch.stack(outputs.scores, dim=0).squeeze(1), dim=-1)
    token_probs = scores[torch.arange(scores.size(0)), newly_generated_ids]
    token_probs = torch.log(token_probs + 1e-10)
    response = tokenizer.decode(newly_generated_ids, skip_special_tokens=True)
    return response, token_probs