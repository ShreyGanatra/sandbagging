from dataclasses import dataclass, field
import torch


@dataclass
class MCQAPredictor:
    """One forward pass, no generation. Restricts next-token logits to the
    choice ids and argmaxes. Cheap — this is why it stays a separate class
    from GenerationPredictor rather than a special case of it."""
    choice_ids: torch.Tensor
    choices: list[str] = field(default_factory=lambda: ["A", "B", "C", "D"])

    @torch.no_grad()
    def __call__(self, model, tokenizer, input_ids, attn_mask, ctx):
        with ctx:
            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
        restricted = logits[:, -1, :][:, self.choice_ids]
        pred_idx = restricted.argmax(dim=-1).tolist()
        return [self.choices[i] for i in pred_idx]


@dataclass
class GenerationPredictor:
    """Autoregressive generation for free-text answers."""
    max_new_tokens: int = 256
    do_sample: bool = False

    @torch.no_grad()
    def __call__(self, model, tokenizer, input_ids, attn_mask, ctx):
        with ctx:
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = out_ids[:, input_ids.shape[1]:]
        return [t.strip() for t in tokenizer.batch_decode(gen_ids, skip_special_tokens=True)]


def build_choice_ids(tokenizer, sample_prompt, choices=("A", "B", "C", "D")):
    """Derive each choice letter's token id as it actually tokenizes after
    a real prompt from your data — never assume a fixed literal like ' A',
    since the char before the answer differs by model (Mistral ends
    '[/INST]', Qwen ends 'assistant\\n')."""
    prefix_ids = tokenizer(sample_prompt, add_special_tokens=False)["input_ids"]
    ids = []
    for c in choices:
        full_ids = tokenizer(f"{sample_prompt}{c}", add_special_tokens=False)["input_ids"]
        new_ids = full_ids[len(prefix_ids):]
        assert len(new_ids) == 1, f"{c!r} tokenizes to {len(new_ids)} tokens, not 1: {new_ids}"
        ids.append(new_ids[0])
    return torch.tensor(ids)