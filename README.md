# Requirements
```bash
conda create -n mixlora_dsi python==3.10.14
pip install -r requirements.txt
```

# Important

Go to ~/miniconda/conda/envs/mixlora_dsi/lib/python3.10/site-packages/transformers/generation/utils.py

Add this following code as global variable at the top of the file:

```python
masks = torch.zeros((8, 32100 + 2048 * 7), dtype=torch.int64)
temp = [
    [i for i in range(32100 + 2048 * j)]
    + [i for i in range(32100 + 2048 * (j + 1), 32100 + 2048 * 8)]
    for j in range(8)
]
for i, mask in enumerate(temp):
    masks[i] = torch.tensor(mask, dtype=torch.int64)
```

Then, in the method _beam_search of class GenerationMixin, add the following code
```python
sequence_length = outputs.logits.shape[1]
    for i in range(sequence_length):
        outputs.logits[:, i, :] = outputs.logits[:, i, :].index_fill_(dim=-1, index=masks[i].to(outputs.logits.device), value=float("-inf"))
```
just before this line: 
```python
next_token_logits = outputs.logits[:, -1, :].clone()
```