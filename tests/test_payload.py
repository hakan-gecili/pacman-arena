"""Shared payload construction, using a whitespace tokenizer in place of Laya's."""
from pacman_arena.game import Game
from pacman_arena.models import nimble_prompt, tev_prompt
from pacman_arena.payload import INSTRUCTIONS, LayaBudget, fair_payload


class WordTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_payload_fits_budget_and_is_shared():
    budget = LayaBudget(WordTokenizer(), {"max_len": 512, "head_max_len": 192})
    payload = fair_payload(Game(seed=1), budget, depth=10, per_direction=3, max_options=10)
    assert not payload.truncated
    assert list(payload.criteria) == list("ABCDEFGHIJ"[:len(payload.options)])
    for render in (tev_prompt, nimble_prompt):
        prompt = render(payload)
        assert INSTRUCTIONS in prompt and payload.state in prompt
        assert all(text in prompt for text in payload.criteria.values())
