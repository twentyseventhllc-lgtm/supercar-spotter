# NOTE: keep torch / open_clip imports OUT of module top level (see Task 4).
from dataclasses import dataclass


@dataclass
class Verdict:
    label: str
    confidence: float
    is_supercar: bool


def pick_best(scores, brands, negatives, margin, min_confidence=0.0):
    brand_scores = {label: scores[label] for label in brands if label in scores}
    if not brand_scores:
        return Verdict(label="", confidence=0.0, is_supercar=False)

    best_label = max(brand_scores, key=brand_scores.get)
    best_brand = brand_scores[best_label]

    neg_scores = [scores[label] for label in negatives if label in scores]
    best_neg = max(neg_scores) if neg_scores else 0.0

    # A catch needs BOTH: it beats the best "ordinary car" score by `margin`,
    # AND its absolute brand confidence clears `min_confidence`. The floor is
    # what kills weak ~50-60% guesses on blurry/ambiguous cars while letting a
    # distinctive supercar (which scores ~0.9+) through.
    beats_margin = (best_brand - best_neg) >= margin
    clears_floor = best_brand >= min_confidence
    is_supercar = beats_margin and clears_floor
    return Verdict(label=best_label, confidence=best_brand, is_supercar=is_supercar)


class BrandClassifier:
    """Zero-shot brand scorer. Lazy-imports heavy deps so `pick_best` stays light."""

    def __init__(self, labels, model_name="ViT-B-32",
                 pretrained="laion2b_s34b_b79k", template="a photo of {}"):
        import torch
        import open_clip

        self._torch = torch
        self.labels = list(labels)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        tokenizer = open_clip.get_tokenizer(model_name)

        prompts = [template.format(label) for label in self.labels]
        tokens = tokenizer(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        self._text_features = text_features

    def score(self, crop_bgr):
        import cv2
        from PIL import Image

        torch = self._torch
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = self.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ self._text_features.T
            probs = logits.softmax(dim=-1).squeeze(0).tolist()

        return {label: float(p) for label, p in zip(self.labels, probs)}
