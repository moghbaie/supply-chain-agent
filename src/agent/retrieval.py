"""Find project policy passages by word overlap. Runs locally and only reads the policy file."""
import math
import re
from collections import Counter
from pathlib import Path

import yaml

POLICY_PATH = Path("configs/agent_policy.yaml")
STOP_WORDS = {"a", "an", "and", "are", "be", "by", "do", "for", "in", "is", "it", "of", "on", "or", "the", "to", "what", "when", "with"}


def words(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower().replace("_", " ")) if w not in STOP_WORDS]


class PolicyIndex:
    """Rank passages with BM25 scores, breaking ties by file order so results repeat."""

    def __init__(self, path=POLICY_PATH):
        self.passages = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["passages"]
        self.counts = [Counter(words(f"{p['title']} {p['text']}")) for p in self.passages]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.frequency = Counter(word for count in self.counts for word in count)

    def search(self, query, limit=2, k=1.5, b=0.75):
        average = sum(self.lengths) / len(self.lengths)
        scored = []
        for position, (count, length) in enumerate(zip(self.counts, self.lengths)):
            score = 0.0
            for word in set(words(query)):
                if count[word]:
                    weight = math.log(1 + (len(self.passages) - self.frequency[word] + 0.5) / (self.frequency[word] + 0.5))
                    score += weight * count[word] * (k + 1) / (count[word] + k * (1 - b + b * length / average))
            if score > 0:
                scored.append((-score, position))
        return [{"id": self.passages[i]["id"], "text": self.passages[i]["text"]} for _, i in sorted(scored)[:limit]]
