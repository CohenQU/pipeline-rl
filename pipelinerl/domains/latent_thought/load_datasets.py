"""
Dataset loader for the latent_thought domain.

Loads raw text from a HuggingFace dataset, builds documents (optionally by
joining short rows into articles), finds candidate (prefix, suffix) cut points
at paragraph boundaries (\\n\\n), filters by length, and returns problem dicts
that keep the full text plus a list of cut offsets. The actual cut is chosen
online at rollout time, so the same row produces different (prefix, suffix)
pairs across visits.

Each problem dict has:
  - "dataset":     source dataset name
  - "id":          sequential id within the loaded list
  - "text":        full text of the document
  - "cut_offsets": list[int] of character offsets that pass the length filter

Row-grouping modes:
  - "row" (default): each row of the dataset becomes one document. Use this when
    rows are already long, multi-paragraph chunks (e.g. CommonCrawl-style).
  - "article": rows whose stripped form matches `article_heading_pattern`
    (default targets the wikitext top-level heading "= Title =") start a new
    article. Non-heading rows are joined by "\\n\\n" into the current article.
    Empty rows are dropped.
  - "auto": pick "article" when most rows are short, else "row".
"""

import io
import json
import logging
import random
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

from datasets import load_dataset
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)


def _stream_jsonl_zst_from_hub(
    hub_id: str,
    text_field: str,
    shuffle_seed: Optional[int],
    skip_rows: int,
    max_rows: Optional[int],
) -> Iterator[Dict[str, Any]]:
    """Stream rows from a HF dataset of `.jsonl.zst` shards, bypassing pyarrow.

    HF datasets uses pyarrow's JSON reader, which infers a per-shard schema
    and crashes when nested fields drift across rows. allenai/dolma3_dolmino
    has this problem in `metadata` (e.g. `metadata/google_gemma-3-12b-it_contains_pii`
    is a number in some rows and a boolean in others). `select_columns` does
    NOT push down to the parser — pyarrow still parses the full JSON line
    before any projection happens. See ERR-002.

    This loader sidesteps pyarrow entirely: it lists the dataset's `.jsonl.zst`
    shards via HfFileSystem, decompresses with zstandard, and parses each
    line with the stdlib `json` module (which is per-row and tolerates schema
    differences). Only `text_field` and `id` are kept from each row.

    Shard ordering is deterministic from `shuffle_seed`. If train and test
    share the same seed, they see the same shard order — apply different
    `skip_rows` / `max_rows` to carve disjoint slices.
    """
    # Imported lazily to avoid hard dep when other datasets are used.
    import zstandard
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    glob_pattern = f"datasets/{hub_id}/**/*.jsonl.zst"
    shards = sorted(fs.glob(glob_pattern))
    if not shards:
        raise ValueError(
            f"No .jsonl.zst shards found under {glob_pattern}. "
            "Loader 'jsonl_zst_hf' requires the HF dataset to ship as compressed JSONL."
        )
    if shuffle_seed is not None:
        rng = random.Random(int(shuffle_seed))
        rng.shuffle(shards)
    logger.info(
        f"[jsonl_zst_hf] {hub_id}: {len(shards)} shards, "
        f"shuffle_seed={shuffle_seed}, skip_rows={skip_rows}, max_rows={max_rows}"
    )

    rows_seen = 0
    rows_yielded = 0
    for shard in shards:
        if max_rows is not None and rows_yielded >= max_rows:
            break
        try:
            with fs.open(shard, "rb") as f:
                dctx = zstandard.ZstdDecompressor()
                with dctx.stream_reader(f) as zs:
                    text_stream = io.TextIOWrapper(zs, encoding="utf-8", errors="replace")
                    for line in text_stream:
                        rows_seen += 1
                        if rows_seen <= skip_rows:
                            continue
                        if max_rows is not None and rows_yielded >= max_rows:
                            break
                        try:
                            row = json.loads(line)
                        except (json.JSONDecodeError, ValueError):
                            continue
                        text = row.get(text_field)
                        if not isinstance(text, str):
                            continue
                        rows_yielded += 1
                        yield {text_field: text, "id": row.get("id")}
        except Exception as e:
            logger.warning(
                f"[jsonl_zst_hf] skipping shard {shard} ({type(e).__name__}: {e})"
            )
            continue
    logger.info(
        f"[jsonl_zst_hf] {hub_id}: yielded {rows_yielded} rows (after skipping {skip_rows})"
    )


_PARAGRAPH_RE = re.compile(r"(?:\r\n|\n){2,}")
# Wikitext top-level article heading: " = Title = " (single = on each side).
# We do not match sub-headings like " = = Section = = ".
_WIKITEXT_HEADING_RE = re.compile(r"^=\s*[^=].*[^=]\s*=$")


def _paragraph_cut_offsets(text: str) -> List[int]:
    """Return character offsets just after each paragraph break.

    Each offset is a candidate cut point: text[:offset] is the prefix,
    text[offset:] is the suffix.
    """
    offsets = []
    for m in _PARAGRAPH_RE.finditer(text):
        offsets.append(m.end())
    return offsets


def _filter_offsets(
    text: str,
    offsets: List[int],
    min_side_chars: int,
    max_total_chars: int,
) -> List[int]:
    n = len(text)
    kept = []
    for off in offsets:
        prefix_len = off
        suffix_len = n - off
        if prefix_len < min_side_chars:
            continue
        if suffix_len < min_side_chars:
            continue
        if prefix_len + suffix_len > max_total_chars:
            # Even with full suffix, prefix is bounded; we still keep but the
            # rollout truncates. To keep memory cheap, we skip rows whose
            # whole text exceeds max_total_chars.
            continue
        kept.append(off)
    return kept


def _process_text_row(row: Dict[str, Any], text_field: str) -> str:
    val = row.get(text_field)
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    return str(val)


def _is_heading(text: str, heading_re: re.Pattern) -> bool:
    return bool(heading_re.match(text.strip()))


def _build_documents_by_article(
    texts: List[str],
    heading_re: re.Pattern,
) -> List[str]:
    """Group consecutive non-heading rows into articles, joined by \\n\\n.

    A row matching `heading_re` starts a new article and is dropped (we don't
    keep the heading text itself, since it's a structural marker rather than
    natural prose).
    """
    documents: List[str] = []
    current: List[str] = []

    def flush() -> None:
        if current:
            documents.append("\n\n".join(current))
            current.clear()

    for raw in texts:
        stripped = raw.strip()
        if not stripped:
            continue
        if _is_heading(stripped, heading_re):
            flush()
            continue
        current.append(stripped)
    flush()
    return documents


def _detect_grouping_strategy(sample_texts: List[str]) -> str:
    """Heuristic: choose "article" if most non-empty rows are short and lack
    paragraph breaks; else "row".
    """
    nonempty = [t for t in sample_texts if t.strip()]
    if not nonempty:
        return "row"
    short = sum(1 for t in nonempty if len(t) < 1500 and "\n\n" not in t)
    return "article" if short >= 0.7 * len(nonempty) else "row"


def add_ids(samples: List[Dict]) -> List[Dict]:
    for i, s in enumerate(samples):
        s.setdefault("id", i)
    return samples


def load_datasets(
    dataset_names: List[str | Dict[str, Any]] | Dict[str, Any] | str | None,
    seed: int | None = None,
    min_side_chars: int = 256,
    max_total_chars: int = 24000,
    text_field: str = "text",
    row_grouping: str = "auto",
    article_heading_pattern: Optional[str] = None,
) -> List[Dict]:
    """Load text rows and pre-compute paragraph cut offsets.

    Each spec in `dataset_names` may be:
      - A dict with keys "hub_id", optional "config", optional "split",
        optional "row_grouping" override, optional "text_field" override.
      - A plain "org/repo" string.

    Documents whose text has no surviving cut candidates are dropped. The full
    text is retained on each problem so the rollout can re-tokenize and
    re-slice against the evaluator tokenizer.
    """
    if dataset_names is None:
        return []

    if isinstance(dataset_names, (DictConfig, ListConfig)):
        dataset_names = OmegaConf.to_container(dataset_names, resolve=True)

    if isinstance(dataset_names, dict):
        dataset_names = [dataset_names]
    elif isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    elif not isinstance(dataset_names, list):
        dataset_names = list(dataset_names)

    heading_re = re.compile(article_heading_pattern) if article_heading_pattern else _WIKITEXT_HEADING_RE

    all_samples: List[Dict] = []

    for spec in dataset_names:
        if isinstance(spec, dict):
            hub_id = spec.get("hub_id")
            if not hub_id:
                raise ValueError("Dataset spec must include a 'hub_id' field.")
            config = spec.get("config")
            split = spec.get("split", "train")
            trust_remote_code = spec.get("trust_remote_code", True)
            spec_text_field = spec.get("text_field", text_field)
            spec_row_grouping = spec.get("row_grouping", row_grouping)
            spec_streaming = bool(spec.get("streaming", False))
            spec_shuffle_seed = spec.get("shuffle_seed", None)
            spec_shuffle_buffer = int(spec.get("shuffle_buffer_size", 10000))
            spec_skip_rows = int(spec.get("skip_rows", 0))
            spec_max_rows = spec.get("max_rows", None)
            spec_keep_columns = spec.get("keep_columns", None)
            spec_loader_kind = spec.get("loader_kind", "hf")
        elif isinstance(spec, str) and "/" in spec:
            hub_id = spec
            config = None
            split = "train"
            trust_remote_code = True
            spec_text_field = text_field
            spec_row_grouping = row_grouping
            spec_streaming = False
            spec_shuffle_seed = None
            spec_shuffle_buffer = 10000
            spec_skip_rows = 0
            spec_max_rows = None
            spec_keep_columns = None
            spec_loader_kind = "hf"
        else:
            logger.warning(f"Unrecognized dataset spec, skipping: {spec!r}")
            continue

        dataset_label = hub_id.split("/")[-1] + (f"/{config}" if config else "") + f":{split}"

        if spec_loader_kind == "jsonl_zst_hf":
            # Custom streaming loader for HF datasets that ship as .jsonl.zst
            # shards (e.g. allenai/dolma3_dolmino_mix-10B-1025). Bypasses HF's
            # pyarrow-based JSON reader to avoid cross-shard schema drift
            # crashes; see _stream_jsonl_zst_from_hub docstring and ERR-002.
            dataset = _stream_jsonl_zst_from_hub(
                hub_id=hub_id,
                text_field=spec_text_field,
                shuffle_seed=spec_shuffle_seed,
                skip_rows=int(spec_skip_rows),
                max_rows=int(spec_max_rows) if spec_max_rows is not None else None,
            )
        elif spec_loader_kind == "hf":
            load_args: Tuple[Any, ...] = (hub_id,)
            if config is not None:
                load_args += (config,)
            dataset = load_dataset(
                *load_args,
                split=split,
                trust_remote_code=trust_remote_code,
                streaming=spec_streaming,
            )

            # Optional column projection. NOTE: this does NOT push down to
            # pyarrow's JSON parser — useful for normal datasets but does NOT
            # rescue datasets with cross-shard schema drift (use
            # loader_kind: jsonl_zst_hf instead). Kept for forward compat.
            if spec_keep_columns:
                dataset = dataset.select_columns(list(spec_keep_columns))

            # Optional shuffle / skip / take. Used for large datasets where
            # we cannot afford to materialize the full corpus in RAM. With
            # matching `shuffle_seed` across two specs (train and test on the
            # same source), `skip_rows` cleanly carves out a disjoint slice.
            if spec_shuffle_seed is not None:
                if spec_streaming:
                    dataset = dataset.shuffle(
                        seed=int(spec_shuffle_seed),
                        buffer_size=spec_shuffle_buffer,
                    )
                else:
                    dataset = dataset.shuffle(seed=int(spec_shuffle_seed))
            if spec_skip_rows > 0:
                if spec_streaming:
                    dataset = dataset.skip(spec_skip_rows)
                else:
                    n = len(dataset)
                    start = min(spec_skip_rows, n)
                    dataset = dataset.select(range(start, n))
            if spec_max_rows is not None:
                n_take = int(spec_max_rows)
                if spec_streaming:
                    dataset = dataset.take(n_take)
                else:
                    dataset = dataset.select(range(min(n_take, len(dataset))))
        else:
            raise ValueError(
                f"Unknown loader_kind={spec_loader_kind!r} for {hub_id}. "
                "Supported: 'hf' (default, uses datasets.load_dataset), "
                "'jsonl_zst_hf' (custom streaming reader for .jsonl.zst HF datasets)."
            )

        all_texts = [_process_text_row(row, spec_text_field) for row in dataset]

        if spec_row_grouping == "auto":
            sample = all_texts[:200]
            chosen = _detect_grouping_strategy(sample)
            logger.info(f"[{dataset_label}] auto-detected row_grouping={chosen}")
        else:
            chosen = spec_row_grouping

        if chosen == "article":
            documents = _build_documents_by_article(all_texts, heading_re)
        elif chosen == "row":
            documents = [t for t in all_texts if t.strip()]
        else:
            raise ValueError(f"Unknown row_grouping: {chosen!r}")

        kept = 0
        skipped_no_cuts = 0
        for doc in documents:
            offsets = _paragraph_cut_offsets(doc)
            offsets = _filter_offsets(doc, offsets, min_side_chars, max_total_chars)
            if not offsets:
                skipped_no_cuts += 1
                continue
            all_samples.append({
                "dataset": dataset_label,
                "text": doc,
                "cut_offsets": offsets,
            })
            kept += 1

        logger.info(
            f"Loaded latent_thought dataset {dataset_label}: "
            f"{len(documents)} docs from {len(all_texts)} rows; "
            f"kept {kept}, dropped {skipped_no_cuts} (no valid paragraph cut)"
        )

    if seed is not None:
        import random
        rng = random.Random(seed)
        rng.shuffle(all_samples)

    return add_ids(all_samples)
