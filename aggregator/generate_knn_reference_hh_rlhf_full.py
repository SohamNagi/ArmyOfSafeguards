"""Generate KNN reference data from Anthropic HH-RLHF with checkpoints.

This script streams the HH-RLHF dataset, runs every safeguard, and writes
reference rows in configurable batches. Progress is persisted so that long
runs can be resumed without losing already processed samples.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from tqdm import tqdm

# Add parent directory to path so we can import aggregator and safeguards
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

# Import aggregator functions
from aggregator import run_all_safeguards


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="Generate KNN reference data with batching and checkpoints.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of samples to write before flushing to disk and checkpointing.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        default=script_dir / "knn_reference_hh_rlhf_full.checkpoint.json",
        help="Path to checkpoint progress for resuming long runs.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=script_dir / "knn_reference_hh_rlhf_full.jsonl",
        help="JSONL file that receives the reference dataset.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load from Anthropic/hh-rlhf.",
    )
    parser.add_argument(
        "--subset",
        default="harmless-base",
        help="Data directory (subset) passed to load_dataset.",
    )
    return parser.parse_args()


def safe_confidence(results: Dict[str, Any], key: str) -> float:
    block = results.get(key, {})
    if isinstance(block, dict) and "error" not in block:
        return float(block.get("confidence", 0.0))
    return 0.0


def build_record(text: str, is_safe: bool, source: str, results: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "text": text,
        "conf_fact": round(safe_confidence(results, "factuality"), 4),
        "conf_tox": round(safe_confidence(results, "toxicity"), 4),
        "conf_sex": round(safe_confidence(results, "sexual"), 4),
        "conf_jb": round(safe_confidence(results, "jailbreak"), 4),
        "is_safe": is_safe,
        "source": source,
    }


def load_checkpoint(path: Path, total: int) -> Tuple[Dict[str, Any], bool]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            print(f"🔁 Resuming from checkpoint at {path}")
            return data, True
        except json.JSONDecodeError:
            print("⚠️ Checkpoint file is corrupted; starting from scratch.")
    return {
        "chosen_index": 0,
        "rejected_index": 0,
        "phase": "chosen",
        "total": total,
        "pending_records": [],
    }, False


def save_checkpoint(path: Path, state: Dict[str, Any], verbose: bool = True) -> None:
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    if verbose:
        print(
            f"💾 Checkpoint saved → {path} "
            f"(chosen={state.get('chosen_index', 0)}, rejected={state.get('rejected_index', 0)})"
        )


def flush_pending_records(state: Dict[str, Any], output_path: Path, verbose: bool = True) -> None:
    pending = state.get("pending_records") or []
    if not pending:
        return
    with open(output_path, "a", encoding="utf-8") as handle:
        for record in pending:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if verbose:
        print(f"➡️  Restored {len(pending)} pending records from the previous run into {output_path}")
    state["pending_records"] = []


class BatchedWriter:
    def __init__(self, output_path: Path, batch_size: int, resume: bool) -> None:
        self.output_path = output_path
        self.batch_size = max(batch_size, 1)
        self.buffer: List[Dict[str, Any]] = []
        self._mode = "a" if resume and output_path.exists() else "w"

    def add(self, record: Dict[str, Any]) -> None:
        self.buffer.append(record)

    def maybe_flush(self) -> bool:
        if len(self.buffer) >= self.batch_size:
            self._flush()
            return True
        return False

    def flush(self) -> bool:
        if not self.buffer:
            return False
        self._flush()
        return True

    def _flush(self) -> None:
        with open(self.output_path, self._mode, encoding="utf-8") as handle:
            for record in self.buffer:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.buffer.clear()
        self._mode = "a"


def process_split(
    dataset,
    text_key: str,
    is_safe: bool,
    source: str,
    desc: str,
    state: Dict[str, Any],
    state_key: str,
    writer: BatchedWriter,
    checkpoint_path: Path,
) -> None:
    total = len(dataset)
    start_idx = min(state.get(f"{state_key}_index", 0), total)
    state.setdefault("pending_records", [])

    if start_idx >= total:
        print(f"⏭️  {desc} already completed according to checkpoint.")
        state[f"{state_key}_complete"] = True
        return

    progress = tqdm(range(start_idx, total), total=total - start_idx, desc=desc)

    try:
        for idx in progress:
            text = dataset[idx][text_key]
            results = run_all_safeguards(text)
            record = build_record(text, is_safe, source, results)
            writer.add(record)

            state["phase"] = state_key
            state[f"{state_key}_index"] = idx + 1
            state["pending_records"].append(record)
            save_checkpoint(checkpoint_path, state, verbose=False)

            if writer.maybe_flush():
                state["pending_records"] = []
                save_checkpoint(checkpoint_path, state, verbose=False)
    except KeyboardInterrupt:
        print(f"\n🛑 Interrupted while processing {desc}.")
        if writer.flush():
            state["pending_records"] = []
        save_checkpoint(checkpoint_path, state)
        raise

    if writer.flush():
        state["pending_records"] = []

    # Still persist the latest index even if nothing new was flushed.
    save_checkpoint(checkpoint_path, state)
    state[f"{state_key}_complete"] = True


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).parent

    print(f"Loading Anthropic/hh-rlhf dataset (subset='{args.subset}', split='{args.split}')...")
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir=args.subset, split=args.split)
    total = len(dataset)
    print(
        f"Processing {total} prompt pairs (~{total * 2} responses) with batch size {args.batch_size}."
    )

    checkpoint_state, resume = load_checkpoint(args.checkpoint_file, total)
    checkpoint_state.setdefault("pending_records", [])
    if resume and checkpoint_state["pending_records"]:
        flush_pending_records(checkpoint_state, args.output_file)
        save_checkpoint(args.checkpoint_file, checkpoint_state, verbose=False)

    writer = BatchedWriter(args.output_file, args.batch_size, resume)

    completed = False
    try:
        process_split(
            dataset=dataset,
            text_key="chosen",
            is_safe=True,
            source="hh-rlhf-harmless-chosen",
            desc="Chosen responses (safe)",
            state=checkpoint_state,
            state_key="chosen",
            writer=writer,
            checkpoint_path=args.checkpoint_file,
        )

        process_split(
            dataset=dataset,
            text_key="rejected",
            is_safe=False,
            source="hh-rlhf-harmless-rejected",
            desc="Rejected responses (unsafe)",
            state=checkpoint_state,
            state_key="rejected",
            writer=writer,
            checkpoint_path=args.checkpoint_file,
        )

        completed = (
            checkpoint_state.get("chosen_index", 0) >= total
            and checkpoint_state.get("rejected_index", 0) >= total
        )
    except KeyboardInterrupt:
        print("Progress saved. Re-run the script to resume from the last checkpoint.")
    finally:
        if writer.flush():
            checkpoint_state["pending_records"] = []
        if completed and args.checkpoint_file.exists():
            args.checkpoint_file.unlink()
            print(f"✅ Completed! Reference dataset saved to {args.output_file}")
            print("Checkpoint file removed because processing finished successfully.")
        elif not completed:
            # Ensure the latest state is persisted when exiting early.
            save_checkpoint(args.checkpoint_file, checkpoint_state)
            print(f"➡️  Partial progress stored in {args.checkpoint_file}")

    print("\n" + "=" * 60)
    print("KNN Reference Data Generation Complete" if completed else "KNN Data Generation Paused")
    print("=" * 60)
    print(f"Reference data file: {args.output_file}")
    print("\nTo load the reference data, use:")
    print("  from aggregator.aggregator import load_knn_reference_data")
    print(f"  load_knn_reference_data('{args.output_file}')")


if __name__ == "__main__":
    main()