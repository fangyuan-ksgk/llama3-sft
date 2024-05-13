# Llama3-SFT
Supervised FineTuning with Llama-3

This repo fixes two issues which blocks people from successfully perfoming supervised fine-tuning on Llama-3
1. DataCollatorForCompletionOnlyLM Bug: in the process of supervised finetuning, we do NOT want the model to memorize the prompt, but only to memorize the response it should provide, given that prompt or query. This means we need to mask out the loss value computed on the query tokens. DataCollatorForCompletionOnlyLM is designed for this purpose. The only issue is that there is a bug in it. Let's look at the source code:
```python
 self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
    if (response_token_ids== batch["labels"][i][idx : idx + len(response_token_ids)].tolist()):
        response_token_ids_start_idx = idx
```
The issue with the code snippet above is that tokenization is done on the entire prompt-completion sequence, and first ids corresponding to the response usually changes with different prefix. (Unless you have "#### Answer" as your response template, which is not the case for many LLM, for instance, Llama3 uses "<|start_header_id|>assistant<|end_header_id|>\n\n" as the response template, which has varying first response token ids according to different prefix)

