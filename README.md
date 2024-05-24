# Llama3-SFT
Adaptive Supervised FineTuning with Llama-3

![image](https://github.com/fangyuan-ksgk/llama3-sft/assets/66006349/c0979819-796d-4d7e-af5f-56e545dca993)

This repo fixes two issues which blocks people from successfully perfoming supervised fine-tuning on Llama-3

1. DataCollatorForCompletionOnlyLM Bug: in the process of supervised finetuning, we do NOT want the model to memorize the prompt, but only to memorize the response it should provide, given that prompt or query. This means we need to mask out the loss value computed on the query tokens. DataCollatorForCompletionOnlyLM is designed for this purpose. The only issue is that there is a bug in it. Let's look at the source code:
```python
self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)

for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
    if (response_token_ids== batch["labels"][i][idx : idx + len(response_token_ids)].tolist()):
        response_token_ids_start_idx = idx
```
The issue with the code snippet above is that tokenization is done on the entire prompt-completion sequence, and first ids corresponding to the response usually changes with different prefix. (Unless you have "#### Answer" as your response template, which is not the case for many LLM, for instance, Llama3 uses "<|start_header_id|>assistant<|end_header_id|>\n\n" as the response template, which has varying first response token ids according to different prefix). Here is the fix
```python
# Token level search is a wrong idea ---> Merging words into token will make this a logical error
input_ids = batch["input_ids"][i]
# Find location on string level
format_prompt = self.tokenizer.decode(input_ids)
idx = format_prompt.find(self.response_template)
prefix = format_prompt[:idx + len(self.response_template)]
suffix = format_prompt[idx + len(self.response_template):]
# Backward propagate to token level | Want the model to predict the next token for us
prefix_tokens = self.tokenizer.tokenize(prefix, add_special_tokens=False)
suffix_tokens = self.tokenizer.tokenize(suffix, add_special_tokens=False)
diff = len(input_ids) - len(prefix_tokens) - len(suffix_tokens)
response_token_ids_start_idx = len(prefix_tokens) + diff
```

2. The Well-known issue for Llama3 tokenizer, here is how you could fix it
```python
if "Meta-Llama-3-" in model_id:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
else:
    tokenizer.pad_token = tokenizer.unk_token
```


