# multimodal_shap.py

from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path
import os
import re
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from base import BaseSHAP, TextVectorizer, ModelBase
from image_utils import BaseSegmentationModel, SegmentationBased

class MultiModalSHAP(BaseSHAP):
    """
    Combine TokenSHAP + PixelSHAP logic:
      - samples = [token_T0, token_T1, ..., label_O0, label_O1, ...]
      - generate combinations over both token and image-object samples
      - reconstruct prompt from token samples included
      - construct manipulated image by hiding objects not included
      - call model.generate(prompt=..., image_path=...)
      - compute similarities with vectorizer and compute Shapley via BaseSHAP
    """
    def __init__(
        self,
        model: ModelBase,
        splitter,
        segmentation_model: Optional[BaseSegmentationModel] = None,
        manipulator: Optional[SegmentationBased] = None,
        vectorizer: Optional[TextVectorizer] = None,
        debug: bool = False,
        temp_dir: str = "example_temp",
        seed: Optional[int] = None
    ):
        super().__init__(model=model, vectorizer=vectorizer, debug=debug)
        self.splitter = splitter
        self.segmentation_model = segmentation_model
        self.manipulator = manipulator
        self.temp_dir = temp_dir
        self.seed = seed
        os.makedirs(self.temp_dir, exist_ok=True)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # state set during analyze()
        self._current_labels = []
        self._current_masks = []
        self._current_boxes = []
        self._last_prompt = None
        self._last_image_path = None

    def plot_importance_ranking(
        self,
        thumbnail_size=60,
        show_values=True,
    ):
        boxes, labels, scores, masks, image = self._detect_objects(self._last_image_path, return_segmentation=True)
        image = cv2.imread(str(self._last_image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.visualizer.plot_importance_ranking(
            shapley_values=self.shapley_values,  
            image=image,
            masks=masks,
            thumbnail_size=60,
            show_values=True
        )
    # ---------- helpers ----------
    def _detect_objects(self, image_path: Union[str, Path]):
        if self.segmentation_model is None:
            self._current_labels = []
            self._current_masks = []
            self._current_boxes = []
            return

        boxes, labels, scores, masks = self.segmentation_model.segment(image_path)
        if boxes is None:
            boxes = []
        if masks is None:
            masks = []
        if labels is None:
            labels = []

        if len(labels) == 0:
            # treat as zero objects (still permitted)
            self._current_labels = []
            self._current_masks = []
            self._current_boxes = []
            return

        self._current_labels = labels
        self._current_masks = masks
        self._current_boxes = boxes

        if self.debug:
            print("Detected image objects:")
            for i, label in enumerate(labels):
                print(f"  {i}: {label}")

    def _token_samples_with_suffix(self, prompt: str) -> List[str]:
        tokens = self.splitter.split(prompt)
        return [f"{tok}_T{idx}" for idx, tok in enumerate(tokens)]

    def _object_samples_with_suffix(self) -> List[str]:
        return [f"{lbl}_O{idx}" for idx, lbl in enumerate(self._current_labels)]

    def _prepare_generate_args(self, content: Any, **kwargs) -> Dict:
        """Prepare arguments for model.generate()"""
        if isinstance(content, dict):
            return {
                "prompt": content.get("prompt", ""),
                "image_path": content.get("image_path", "")
            }
        return {
            "prompt": kwargs.get("prompt", ""),
            "image_path": str(content)
        }

    # ---------- core overrides ----------
    def _get_samples(self, content: Any) -> List[str]:
        """
        content is expected to be a dict with keys:
          - 'prompt': str
          - 'image_path': Optional[str]
        Returns list of samples like ['the_T0', 'dog_T1', 'person_O0', ...]
        """
        if isinstance(content, dict):
            prompt = content.get("prompt", "")
            image_path = content.get("image_path", None)
        else:
            raise ValueError("_get_samples expects dict with 'prompt' and optional 'image_path'")

        # tokens
        token_samples = self._token_samples_with_suffix(prompt)

        # objects
        if image_path:
            self._detect_objects(image_path)
        else:
            self._current_labels = []
            self._current_masks = []
            self._current_boxes = []

        object_samples = self._object_samples_with_suffix()

        samples = token_samples + object_samples
        if self.debug:
            print(f"_get_samples -> tokens: {len(token_samples)}, objects: {len(object_samples)}, total: {len(samples)}")
        return samples

    def _prepare_combination_args(self, combination: List[str], original_content: Any) -> Dict:
        """
        combination: list of sample strings included in this combo (e.g. ['the_T0','dog_T1','person_O0'])
        original_content: dict with 'prompt' and 'image_path'
        Returns args dict for model.generate(prompt=..., image_path=...)
        """
        if isinstance(original_content, dict):
            prompt = original_content.get("prompt", "")
            image_path = original_content.get("image_path", None)
        else:
            raise ValueError("_prepare_combination_args expects dict original_content")

        # Rebuild prompt from tokens included in combination
        token_parts = [s for s in combination if s.endswith("_T" + s.split("_T")[-1])]
        # extract token text (everything before the _T{idx} suffix)
        tokens_text = [re.sub(r"_T\d+$", "", s) for s in token_parts]
        new_prompt = self.splitter.join(tokens_text)

        # Recreate image by hiding objects NOT included
        # If no segmentation/manipulator provided, keep image_path unchanged
        if image_path and self.segmentation_model is not None and self.manipulator is not None:
            # Ensure objects were detected (should have been in _get_samples call)
            # Build list of objects to keep (indices)
            keep_object_indices = []
            for s in combination:
                if "_O" in s:
                    try:
                        idx = int(s.split("_O")[-1])
                        keep_object_indices.append(idx)
                    except Exception:
                        continue

            # load image and manipulate: hide objects not in keep_object_indices
            import cv2
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            modified = image.copy()

            all_objs = [f"{lbl}_O{i}" for i, lbl in enumerate(self._current_labels)]
            # compute objects to hide (all indices not in keep_object_indices)
            hide_indices = [i for i in range(len(self._current_labels)) if i not in keep_object_indices]

            # call manipulator for each hide index
            for idx in hide_indices:
                modified = self.manipulator.manipulate(
                    modified,
                    self._current_masks,
                    idx,
                    preserve_indices=keep_object_indices
                )

            # write temp file
            temp_name = f"temp_combo_{','.join([str(i) for i in keep_object_indices]) or 'none'}.jpg"
            temp_path = os.path.join(self.temp_dir, temp_name)
            # convert back to BGR for cv2.imwrite
            cv2.imwrite(temp_path, cv2.cvtColor(modified, cv2.COLOR_RGB2BGR))
            final_image_path = temp_path
        else:
            # no manipulation available: pass original image path (or None)
            final_image_path = image_path

        args = {"prompt": new_prompt, "image_path": final_image_path}
        if self.debug:
            print("\n_prepare_combination_args:")
            print(f"  prompt (len words): {len(tokens_text)} -> '{new_prompt}'")
            print(f"  image_path: {final_image_path}")
        return args

    def _get_combination_key(self, combination: List[str], indexes: Tuple[int, ...]) -> str:
        # readable unique key
        return "combo_" + ",".join(str(i) for i in indexes)

    def _generate_essential_combinations(self, samples: List[str]) -> List[Tuple[List[str], Tuple[int, ...]]]:
        n = len(samples)
        essential = []
        # essential combos = all combos missing exactly one sample (so each sample excluded once)
        for i in range(n):
            combo = samples[:i] + samples[i+1:]
            idxs = tuple([j+1 for j in range(n) if j != i])  # 1-based indexes consistent with your other code
            essential.append((combo, idxs))
        return essential

    def _generate_random_combinations(self, samples: List[str], k: int, forbidden_idx_set: set) -> List[Tuple[List[str], Tuple[int, ...]]]:
        """
        Randomly sample additional combinations (not in forbidden_idx_set). Keep combos non-empty.
        """
        n = len(samples)
        combos = set()
        tries = 0
        max_tries = max(100000, k * 100)
        while len(combos) < k and tries < max_tries:
            tries += 1
            # build a random subset: ensure not empty
            included = []
            idxs = []
            for idx in range(n):
                if random.random() < 0.5:
                    included.append(samples[idx])
                    idxs.append(idx + 1)
            if len(included) == 0:
                # ensure non-empty: pick one random sample
                i = random.randrange(n)
                included = [samples[i]]
                idxs = [i + 1]
            idxs_tuple = tuple(sorted(idxs))
            if idxs_tuple in forbidden_idx_set:
                continue
            combos.add((tuple(included), idxs_tuple))
        # convert to the required list-of-tuples type (list, tuple)
        return [(list(c), idxs) for (c, idxs) in combos]

    def _get_result_per_combination(self, content: Any, sampling_ratio: float, max_combinations: Optional[int] = None):
        """
        Similar logic to PixelSHAP._get_result_per_combination but for the full multimodal sample list.
        Returns dict: key -> (response, indexes, combination)
        """
        samples = self._get_samples(content)
        n = len(samples)
        if n == 0:
            raise ValueError("No samples found (no tokens and no objects).")

        if self.debug:
            print(f"Total samples: {n}")

        # essential combinations
        essential = self._generate_essential_combinations(samples)
        essential_set = set(idxs for (_, idxs) in essential)
        num_essential = len(essential)

        if max_combinations is not None and max_combinations < num_essential:
            if self.debug:
                print(f"max_combinations ({max_combinations}) < essential combos ({num_essential}); using all essential combos.")

            max_combinations = num_essential

        remaining_budget = float('inf')
        if max_combinations is not None:
            remaining_budget = max(0, (max_combinations - num_essential))

        # sampling ratio logic
        if sampling_ratio < 1.0:
            theoretical_total = 2 ** n - 1
            theoretical_additional = max(0, theoretical_total - num_essential)
            desired_additional = int(theoretical_additional * sampling_ratio)
            num_additional = min(desired_additional, remaining_budget)
        else:
            # sample all combos up to remaining budget
            num_additional = int(remaining_budget) if remaining_budget != float('inf') else 0

        if self.debug:
            print(f"Essential combos: {num_essential}, Additional to sample: {num_additional}")

        additional = []
        if num_additional > 0:
            additional = self._generate_random_combinations(samples, num_additional, essential_set)

        all_combos = essential + additional
        responses = {}

        for idx, (combo, idxs) in enumerate(tqdm(all_combos, desc="Processing combinations")):

            # Optional: deterministic per-combination seed
            call_seed = None
            if self.seed is not None:
                call_seed = self.seed + idx

            args = self._prepare_combination_args(combo, content)

            # Pass seed ONLY if supported
            if call_seed is not None:
                try:
                    response = self.model.generate(**args, seed=call_seed)
                except TypeError:
                    response = self.model.generate(**args)
            else:
                response = self.model.generate(**args)

            key = self._get_combination_key(combo, idxs)
            responses[key] = (response, idxs, list(combo))

        return responses

    def _get_df_per_combination(self, responses: Dict[str, Tuple[str, Tuple[int, ...], List[str]]], baseline_text: str) -> pd.DataFrame:
        rows = []
        # all sample labels (to compute hidden list) -> need full set used in detection
        all_samples = []
        # tokens original (no suffix)
        token_samples = self._token_samples_with_suffix(self._last_prompt or "")
        object_samples = [f"{lbl}_O{i}" for i, lbl in enumerate(self._current_labels)]
        all_samples = token_samples + object_samples

        for key, (response, indexes, combination) in responses.items():
            shown = combination
            hidden = [s for s in all_samples if s not in shown]
            rows.append({
                "Combination_Key": key,
                "Used_Combination": shown,
                "Hidden_Samples": hidden,
                "Response": response,
                "Indexes": indexes
            })
        df = pd.DataFrame(rows)

        # compute similarities between baseline_text and each response via vectorizer
        texts = [baseline_text] + df["Response"].tolist()
        vectors = self.vectorizer.vectorize(texts)
        base_vec = vectors[0]
        compare = vectors[1:]

        # cosine similarity
        def cos_sim(a, b):
            a = a.astype(np.float64)
            b = b.astype(np.float64)
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0:
                return 0.0
            return float(np.dot(a, b) / denom)

        similarities = [cos_sim(base_vec, v) for v in compare]
        df["Similarity"] = similarities

        if self.debug:
            print(f"Built DataFrame with {len(df)} rows; sample similarities head:\n{df[['Combination_Key','Similarity']].head()}")

        return df
    
    def print_baseline(self):
        """
        Print the stored baseline text response generated during analyze().
        """
        if not hasattr(self, "baseline_text") or self.baseline_text is None:
            print("Baseline text not computed yet. Run analyze() first.")
            return

        print("\n===== BASELINE TEXT RESPONSE =====\n")
        print(self.baseline_text)
        print("\n==================================\n")

    def analyze(
        self,
        prompt: str,
        image_path: Optional[Union[str, Path]] = None,
        sampling_ratio: float = 0.5,
        max_combinations: Optional[int] = None,
        print_highlight_text: bool = False,
        cleanup_temp_files: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Perform full multimodal SHAP analysis.
        """
        if self.debug:
            print("Starting MultiModalSHAP analyze()")
            if image_path:
                print(f"Image: {image_path}")
            print(f"Prompt: {prompt}")

        self._last_prompt = prompt.strip()
        self._last_image_path = str(image_path) if image_path else None

        content = {"prompt": self._last_prompt, "image_path": self._last_image_path}

        # baseline
        self.baseline_text = self._calculate_baseline(content)

        # responses for combinations
        responses = self._get_result_per_combination(content, sampling_ratio=sampling_ratio, max_combinations=max_combinations)

        # build df
        self.results_df = self._get_df_per_combination(responses, self.baseline_text)

        # compute shapley
        raw_shapley = self._calculate_shapley_values(self.results_df, content)

        # fix keys: remove suffixes where obvious (e.g., 'dog_O0' -> 'dog' if no duplicate label conflict)
        fixed = {}
        # gather label-only counts to detect duplicates (like two 'person' objects)
        label_counts = {}
        for s in (self._token_samples_with_suffix(self._last_prompt) + self._object_samples_with_suffix()):
            # token: "word_T0" -> key "word", object: "dog_O0" -> "dog"
            base = re.sub(r"_T\d+$", "", s)
            base = re.sub(r"_O\d+$", "", base)
            label_counts[base] = label_counts.get(base, 0) + 1

        for key, val in raw_shapley.items():
            # raw keys are built from responses' Used_Combination entries when BaseSHAP expects them; but they may contain suffixes
            # try to remove trailing numeric-only suffix like "_0" that some pipelines produce
            candidate = re.sub(r"_\d+$", "", key)
            # If candidate collides and label_counts[candidate] > 1, keep original key
            if candidate in label_counts and label_counts[candidate] == 1:
                fixed[candidate] = val
            else:
                fixed[key] = val

        self.shapley_values = fixed

        if print_highlight_text and hasattr(self, "shapley_values"):
            # simple highlight of tokens portion if requested (reuse TokenSHAP style)
            try:
                token_values = {k: v for k, v in self.shapley_values.items() if not k.endswith("_O")}
                # naive print
                for token, value in token_values.items():
                    print(f"{token}: {value:.4f}")
            except Exception:
                pass

        if cleanup_temp_files:
            # remove temp image files created in temp_dir
            for f in os.listdir(self.temp_dir):
                if f.startswith("temp_combo") or f.startswith("temp_"):
                    try:
                        os.remove(os.path.join(self.temp_dir, f))
                    except Exception:
                        pass

        return self.results_df, self.shapley_values
    
    def rerun_with_top_pairs(
            self,
            top_k_text: int = 3,
            top_k_objects: int = 3,
            n_iterations: int = 3,
            max_combinations: Optional[int] = None,
            cleanup_temp_files: bool = True,
            debug: bool = True
        ):
        import re, cv2, itertools, os, hashlib, copy

        if not hasattr(self, "shapley_values"):
            raise ValueError("Run analyze() first.")

        stage2_dir = os.path.join(self.temp_dir, "stage2_pairs")
        os.makedirs(stage2_dir, exist_ok=True)

        # ------------------ Select top features ------------------
        text_shap = {k: v for k, v in self.shapley_values.items() if "_T" in k}
        obj_shap  = {k: v for k, v in self.shapley_values.items() if "_O" in k}

        selected_text = [k for k,_ in sorted(text_shap.items(), key=lambda x:abs(x[1]), reverse=True)[:top_k_text]]
        selected_obj  = [k for k,_ in sorted(obj_shap.items(),  key=lambda x:abs(x[1]), reverse=True)[:top_k_objects]]

        pair_features = list(itertools.product(selected_text, selected_obj))

        # ------------------ Helpers ------------------
        def strip_t_suffix(s):
            return re.sub(r"_T\d+(?:_\d+)*$", "", s)

        def extract_o_index(s):
            m = re.search(r"_O(\d+)", s)
            return int(m.group(1)) if m else None

        all_tokens = self._token_samples_with_suffix(self._last_prompt)
        fixed_tokens = [strip_t_suffix(t) for t in all_tokens]

        # ------------------ Override sampler ------------------
        def restricted_get_samples(_):
            return pair_features
        self._get_samples = restricted_get_samples

        # ------------------ Patch args builder ------------------
        def patched_prepare_args(combination, original_content):

            masked_tokens = set()
            keep_objects = set()

            for (t, o) in combination:
                masked_tokens.add(strip_t_suffix(t))
                idx = extract_o_index(o)
                if idx is not None:
                    keep_objects.add(idx)

            remaining = [t for t in fixed_tokens if t not in masked_tokens]
            new_prompt = self.splitter.join(remaining)

            final_path = original_content.get("image_path")

            if final_path and self.manipulator and self.segmentation_model:

                image = cv2.imread(str(final_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                modified = image.copy()

                all_indices = set(range(len(self._current_labels)))
                hide_indices = list(all_indices - keep_objects)

                for idx in hide_indices:
                    modified = self.manipulator.manipulate(
                        modified,
                        self._current_masks,
                        idx,
                        preserve_indices=list(keep_objects)
                    )

                combo_hash = hashlib.md5(
                    ("|".join(sorted(map(str, combination)))).encode()
                ).hexdigest()[:10]

                iter_id = getattr(self, "_iter_id", 0)

                fname = f"pair_combo_{combo_hash}_iter{iter_id}.jpg"
                temp_path = os.path.join(stage2_dir, fname)

                cv2.imwrite(temp_path, cv2.cvtColor(modified, cv2.COLOR_RGB2BGR))
                final_path = temp_path

            if debug:
                all_pairs = set(pair_features)
                masked_pairs = sorted(all_pairs - set(combination))

                print("\nPAIR SHAP DEBUG (MASKED)")
                for p in masked_pairs:
                    print(" ", p)

                annotated = []
                masked_text = [strip_t_suffix(t) for (t, _) in masked_pairs]

                for tok in self._token_samples_with_suffix(self._last_prompt):
                    base = strip_t_suffix(tok)
                    annotated.append(f"[MASK:{base}]" if base in masked_text else base)

                print("Prompt:", self.splitter.join(annotated))
                print("Image:", final_path)
                print("-" * 60)

            return {"prompt": new_prompt, "image_path": final_path}

        self._prepare_combination_args = patched_prepare_args

        # ------------------ Monte Carlo wrapper ------------------
        all_results = []

        for i in range(n_iterations):
            self._iter_id = i

            res = self._get_result_per_combination(
                content={"prompt": self._last_prompt, "image_path": self._last_image_path},
                sampling_ratio=1.0,
                max_combinations=max_combinations
            )

            all_results.append(res)

        # ------------------ Merge results ------------------
        merged_results = {}
        for res in all_results:
            merged_results.update(res)

        # Convert merged results into DataFrame
        df = self._get_df_per_combination(merged_results, self.baseline_text)

        pair_shap = self._calculate_shapley_values(
            df,
            {"prompt": self._last_prompt, "image_path": self._last_image_path}
        )

        # ------------------ Cleanup ------------------
        if cleanup_temp_files:
            for f in os.listdir(stage2_dir):
                if f.startswith("pair_combo_"):
                    os.remove(os.path.join(stage2_dir, f))

        return df, pair_shap

    def compute_combined_importance(self, top_k: int = 10):
        """
        Computes and ranks importance for text tokens, objects, and their combined influence.
        Returns three DataFrames: text_df, object_df, combined_df.
        """
        if not hasattr(self, "shapley_values") or not self.shapley_values:
            raise ValueError("Run analyze() first to compute shapley_values.")

        # Separate text and object shap values
        text_shap = {k: v for k, v in self.shapley_values.items() if "_T" in k}
        obj_shap = {k: v for k, v in self.shapley_values.items() if "_O" in k}

        # Convert to DataFrames
        text_df = pd.DataFrame([
            {"type": "text", "feature": k, "base": re.sub(r"_T\d+$", "", k), "shap_value": v}
            for k, v in text_shap.items()
        ]).sort_values("shap_value", ascending=False)

        obj_df = pd.DataFrame([
            {"type": "object", "feature": k, "base": re.sub(r"_O\d+$", "", k), "shap_value": v}
            for k, v in obj_shap.items()
        ]).sort_values("shap_value", ascending=False)

        # Compute all text-object pairs with a combined importance metric
        combined_rows = []
        for t_key, t_val in text_shap.items():
            t_base = re.sub(r"_T\d+$", "", t_key)
            for o_key, o_val in obj_shap.items():
                o_base = re.sub(r"_O\d+$", "", o_key)
                # Combine importance as geometric mean (balanced scale)
                combined_score = np.sqrt(abs(t_val) * abs(o_val))
                combined_rows.append({
                    "text_token": t_base,
                    "object_label": o_base,
                    "combined_importance": combined_score,
                    # "text_shap": t_val,
                    # "object_shap": o_val
                })

        combined_df = pd.DataFrame(combined_rows)
        combined_df = combined_df.sort_values("combined_importance", ascending=False)

        # Print summary of top features
        print("\nTop text features:")
        print(text_df.head(top_k))
        print("\nTop object features:")
        print(obj_df.head(top_k))
        print("\nTop combined text-object pairs:")
        print(combined_df.head(top_k))

        return text_df, obj_df, combined_df
