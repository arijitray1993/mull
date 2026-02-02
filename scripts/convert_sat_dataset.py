#!/usr/bin/env python
"""
Script to convert the SAT dataset to a format compatible with newer datasets/pyarrow versions.

Run this script in the llava_dino environment (Python 3.10, datasets==3.0.2):
    conda activate llava_dino
    python scripts/convert_sat_dataset.py

This will:
1. Load the SAT dataset from HuggingFace (array/SAT)
2. Convert image_bytes to proper PIL Image format using datasets Image feature
3. Save and re-upload to HuggingFace as array/SAT-v2

Dataset structure:
- image_bytes: list of bytes (images encoded as bytes)
- question: string
- answers: list of strings
- correct_answer: string
"""

import os
from datasets import load_dataset, Dataset, DatasetDict, Features, Value, Sequence, Image as ImageFeature
from PIL import Image
import io
from tqdm import tqdm


def convert_and_save_dataset(
    output_dir="/projectnb/ivc-ml/Datasets/MLM_datasets/SAT_converted",
    push_to_hub=False,
    hub_repo_name="array/SAT-v2"
):
    print("Loading SAT dataset from HuggingFace...")
    dataset = load_dataset("array/SAT", batch_size=1)

    print(f"Dataset splits: {dataset.keys()}")

    os.makedirs(output_dir, exist_ok=True)

    converted_splits = {}

    for split_name, split_data in dataset.items():
        print(f"\nProcessing {split_name} split ({len(split_data)} examples)...")
        print(f"Columns: {split_data.column_names}")

        # Check first example
        if len(split_data) > 0:
            first_example = split_data[0]
            print(f"First example keys: {list(first_example.keys())}")

        # Convert image_bytes to PIL Images
        def convert_example(example):
            # image_bytes may already be PIL Images or actual bytes
            images = []
            for im in example.get("image_bytes", []):
                if isinstance(im, Image.Image):
                    # Already a PIL Image
                    img = im
                else:
                    # Raw bytes - decode with PIL
                    img = Image.open(io.BytesIO(im))
                # Convert to RGB if needed (some images might be RGBA or grayscale)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                images.append(img)

            return {
                "images": images,  # Will be encoded by ImageFeature
                "question": example["question"],
                "answers": example["answers"],
                "correct_answer": example["correct_answer"],
                "question_type": example.get("question_type", ""),
            }

        print(f"Converting {split_name} split...")
        converted_data = []
        for i in tqdm(range(len(split_data)), desc=f"Converting {split_name}"):
            converted_data.append(convert_example(split_data[i]))

        # Create new dataset with proper Image feature
        features = Features({
            "images": Sequence(ImageFeature()),
            "question": Value("string"),
            "answers": Sequence(Value("string")),
            "correct_answer": Value("string"),
            "question_type": Value("string"),
        })

        converted_dataset = Dataset.from_list(converted_data, features=features)
        converted_splits[split_name] = converted_dataset
        print(f"Converted {split_name}: {len(converted_dataset)} examples")

    # Create DatasetDict
    final_dataset = DatasetDict(converted_splits)

    # Save to disk
    print(f"\nSaving converted dataset to {output_dir}...")
    final_dataset.save_to_disk(output_dir)
    print("Saved to disk!")

    # Optionally push to hub
    if push_to_hub:
        print(f"\nPushing to HuggingFace Hub as {hub_repo_name}...")
        final_dataset.push_to_hub(hub_repo_name)
        print(f"Uploaded to {hub_repo_name}!")

    print(f"\nDataset conversion complete!")
    print(f"\nTo load the converted dataset:")
    print(f"  from datasets import load_from_disk")
    print(f"  dataset = load_from_disk('{output_dir}')")
    print(f"\nTo upload to HuggingFace manually:")
    print(f"  dataset.push_to_hub('{hub_repo_name}')")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert SAT dataset to compatible format")
    parser.add_argument("--output_dir", default="/projectnb/ivc-ml/Datasets/MLM_datasets/SAT_converted",
                        help="Directory to save converted dataset")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push converted dataset to HuggingFace Hub")
    parser.add_argument("--hub_repo_name", default="array/SAT-v2",
                        help="HuggingFace Hub repo name for upload")
    args = parser.parse_args()

    convert_and_save_dataset(
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        hub_repo_name=args.hub_repo_name
    )
