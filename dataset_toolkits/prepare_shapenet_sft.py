#!/usr/bin/env python3
"""Build ShapeNet mesh conversations from descriptions, images, and mesh tokens."""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List


shapenet_classes = {
    "chair": "03001627",
    "table": "04379243",
    "airplane": "02691156",
    "car": "02958343",
    "rifle": "04090263",
}

img_template_path = Path(__file__).resolve().parents[1] / "data/templates/image_to_3d.json"
text_template_path = Path(__file__).resolve().parents[1] / "data/templates/text_to_3d.json"
understanding_template_path = Path(__file__).resolve().parents[1] / "data/templates/understanding.json"

with open(img_template_path, "r", encoding="utf-8") as f:
    img_templates = json.load(f)

with open(text_template_path, "r", encoding="utf-8") as f:
    text_templates = json.load(f)

with open(understanding_template_path, "r", encoding="utf-8") as f:
    understanding_templates = json.load(f)


def load_description_json(json_path: str) -> List[Dict[str, Any]]:
    """Load mesh descriptions from description.json."""
    with open(json_path, "r", encoding="utf-8") as f:
        descriptions = json.load(f)
    return descriptions


def read_token_sequence(txt_path: str) -> str | None:
    """Read a byte sequence file and format it as mesh tokens."""
    if not os.path.exists(txt_path):
        return None

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        token_sequence = convert_to_mesh_tokens(content)
        return token_sequence
    except Exception as e:
        print(f"Failed to read token file: {txt_path}, error: {e}")
        return None


def convert_to_mesh_tokens(content: str) -> str | None:
    """Wrap integer byte values in mesh tokens and sequence delimiters."""
    if not content:
        return None

    try:
        integers = content.split()

        mesh_tokens = []
        for integer_str in integers:
            integer_str = integer_str.strip()
            if integer_str:
                try:
                    int(integer_str)
                    mesh_tokens.append(f"<mesh{integer_str}>")
                except ValueError:
                    print(f"Warning: non-integer token: {integer_str}")
                    continue

        if mesh_tokens:
            full_sequence = "<mesh_bos>" + "".join(mesh_tokens) + "<mesh_eos>"
            return full_sequence
        else:
            return None
    except Exception as e:
        print(f"Failed to convert token sequence: {e}")
        return None


def generate_conversation(description: str, token_sequence: str, dataset_type: str) -> Dict[str, Any]:
    """Create a mesh conversation from the selected prompt template."""
    # Randomly select a template
    if dataset_type == "description":
        template = random.choice(text_templates)
    elif dataset_type == "understanding":
        template = random.choice(understanding_templates)

    # Deep copy the template to avoid modifying the original
    conversation = json.loads(json.dumps(template))

    # Replace placeholders in messages
    for message in conversation["messages"]:
        if message["role"] == "assistant":
            # Replace #response# placeholder with actual token sequence
            if dataset_type == "description":
                message["content"] = message["content"].replace("#response#", token_sequence)
            elif dataset_type == "understanding":
                message["content"] = message["content"].replace("#object_name#", description)
        elif message["role"] == "user":
            # Replace #object_name# placeholder with actual description
            if dataset_type == "description":
                message["content"] = message["content"].replace("#object_name#", description)
            elif dataset_type == "understanding":
                message["content"] = message["content"].replace("#response#", token_sequence)

    return conversation


def generate_image_conversation(token_sequence: str, image_paths: List[str]) -> Dict[str, Any]:
    """Create an image-conditioned conversation from a randomly selected template."""
    # Load templates if not already loaded
    # Randomly select a template
    template = random.choice(img_templates)

    # Deep copy the template to avoid modifying the original
    conversation = json.loads(json.dumps(template))

    # Replace placeholders in messages
    for message in conversation["messages"]:
        if message["role"] == "assistant":
            # Replace #response# placeholder with actual token sequence
            message["content"] = message["content"].replace("#response#", token_sequence)

    # Replace image path placeholder
    if isinstance(conversation["images"], str) and conversation["images"] == "#image_path#":
        conversation["images"] = image_paths
    elif isinstance(conversation["images"], list):
        # If images is a list, replace each #image_path# placeholder
        conversation["images"] = image_paths

    return conversation


def get_images_from_folder(folder_path: str, num_images: int = 1) -> List[str]:
    """Randomly select up to the requested number of images from a directory."""
    if not os.path.exists(folder_path):
        return None

    image_extensions = [".png", ".jpg", ".jpeg", ".bmp"]
    all_files = os.listdir(folder_path)
    image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in image_extensions]

    if len(image_files) < 1:
        return None

    selected_image = random.choice(image_files)

    image_paths = [os.path.join(folder_path, selected_image)]
    return image_paths


def main():
    parser = argparse.ArgumentParser(description="Prepare ShapeNet SFT conversations.")
    parser.add_argument("--category", choices=tuple(shapenet_classes), required=True)
    parser.add_argument("--description-json", required=True)
    parser.add_argument("--token-dir", required=True, help="Category directory containing <name>/<name>_bytes.txt.")
    parser.add_argument("--render-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", choices=("description", "understanding", "image"), required=True)
    parser.add_argument("--test-json", help="Existing held-out IDs to exclude from training.")
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--max-file-size", type=int, default=1024**3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.test_json and not Path(args.test_json).is_file():
        parser.error("--test-json does not exist; refusing to silently create another split.")
    random.seed(args.seed)
    mesh_type = args.category
    description_json_path = args.description_json
    img_path = args.render_dir
    base_dir = args.token_dir
    output_dir = args.output_dir
    dataset_type = args.task
    limit = args.limit
    max_file_size = args.max_file_size
    test_json_path = args.test_json

    os.makedirs(output_dir, exist_ok=True)

    if dataset_type == "description" or dataset_type == "understanding":
        print("Loading description.json...")
        descriptions = load_description_json(description_json_path)
        print(f"Loaded {len(descriptions)} description entries")

        total_items = len(descriptions)
        processed_items = 0
        skipped_items = 0

        all_valid_items = []
        print("Collecting valid items...")
        for item in descriptions:
            if len(all_valid_items) >= limit:
                print(f"Collected {len(all_valid_items)} valid items; reached the collection limit")
                break

            name = item.get("name", "").strip()
            description_text = item.get("description", "").strip().lower()

            if not name or not description_text:
                skipped_items += 1
                print(f"Skipping item {processed_items + skipped_items}: name or description is empty")
                continue

            subfolder_path = os.path.join(base_dir, name)
            txt_filename = f"{name}_bytes.txt"
            txt_path = os.path.join(subfolder_path, txt_filename)

            if not os.path.exists(txt_path):
                skipped_items += 1
                print(f"Skipping item {processed_items + skipped_items}: token file does not exist - {txt_path}")
                continue

            token_sequence = read_token_sequence(txt_path)
            if not token_sequence:
                skipped_items += 1
                print(f"Skipping item {processed_items + skipped_items}: token sequence is empty")
                continue

            processed_items += 1

            all_valid_items.append({"name": name, "description": description_text, "token_sequence": token_sequence})

            if processed_items % 100 == 0:
                print(f"Collected {processed_items} valid items")

        print(f"Collection complete: {len(all_valid_items)} valid items")

        print("Stage 2: splitting training and test data...")

        if test_json_path and os.path.exists(test_json_path):
            # Exclude the supplied test split from training to prevent data leakage.
            print(f"Loading test split: {test_json_path}")
            with open(test_json_path, "r", encoding="utf-8") as f:
                test_json_data = json.load(f)

            test_names = set()
            for t_item in test_json_data:
                if "name" in t_item:
                    test_names.add(t_item["name"])

            print(f"Test split contains {len(test_names)} items")

            train_items = [item for item in all_valid_items if item["name"] not in test_names]
            test_items = []  # Reuse the supplied test split.

            print(f"Training split: {len(train_items)} items (held-out test items excluded)")
            print(f"Excluded {len(all_valid_items) - len(train_items)} held-out test items")
        else:
            # Create a random split when no existing test split is available.
            if test_json_path:
                print(f"Warning: test split file does not exist - {test_json_path}; using a random split")
            print("Randomly splitting training and test data...")
            random.shuffle(all_valid_items)

            split_index = int(len(all_valid_items) * 0.9)
            train_items = all_valid_items[:split_index]
            test_items = all_valid_items[split_index:]

            print(f"Training split: {len(train_items)} items")
            print(f"Test split: {len(test_items)} items")

        print("Stage 3: generating training conversations...")
        current_data = []
        file_counter = 1
        total_conversations = 0

        def save_current_file():
            """Write the current conversation shard and clear its buffer."""
            nonlocal file_counter, current_data
            if not current_data:
                return

            output_path = f"{output_dir}/llm_shapenet_dataset_{mesh_type}_{file_counter}_{dataset_type}.json"
            print(f"Saving results to {output_path}...")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(current_data, f, ensure_ascii=False, indent=2)

            file_size = os.path.getsize(output_path)
            print(f"File {output_path} size: {file_size / (1024 * 1024):.2f} MB")

            file_counter += 1
            current_data = []

        def check_file_size_and_save():
            """Flush the current shard when its estimated size approaches the limit."""
            if not current_data:
                return False

            # Estimate the serialized JSON size before writing a shard.
            estimated_size = len(json.dumps(current_data, ensure_ascii=False, indent=2).encode("utf-8"))

            if estimated_size >= max_file_size * 0.9:  # Leave 10% headroom for the final serialized shard.
                save_current_file()
                return True
            return False

        for item in train_items:
            conversation = generate_conversation(item["description"], item["token_sequence"], dataset_type)
            current_data.append(conversation)
            total_conversations += 1

            if total_conversations % 100 == 0:
                check_file_size_and_save()
                print(f"Generated {total_conversations} training conversations")

        if current_data:
            save_current_file()

        print(f"Training data complete: {total_conversations} conversations saved in {file_counter - 1} files")

        # Write a test split only when one was not supplied.
        if test_items:
            print("Stage 4: generating test data...")
            test_data = []

            for item in test_items:
                if dataset_type == "description":
                    test_data.append({"name": item["name"], "description": item["description"]})
                else:
                    test_data.append({"name": item["name"], "token_sequence": item["token_sequence"]})

            test_output_path = f"{output_dir}/llm_shapenet_test_{mesh_type}_{dataset_type}.json"
            print(f"Saving test split to {test_output_path}...")
            with open(test_output_path, "w", encoding="utf-8") as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)

            test_file_size = os.path.getsize(test_output_path)
            print(f"Test split file size: {test_file_size / (1024 * 1024):.2f} MB")
        else:
            print("Reusing the supplied test split.")

        print("\n=== Processing complete ===")
        print(f"Total items: {total_items}")
        print(f"Processed items: {processed_items}")
        print(f"Skipped items: {skipped_items}")
        print(f"Training conversations: {total_conversations}")
        print(f"Training shards: {file_counter - 1}")
        if test_items:
            print(f"Test items: {len(test_items)}")
        else:
            print("Test split: using the supplied file")
        print(f"Output directory: {output_dir}")

    elif dataset_type == "image":
        processed_items = 0
        skipped_items = 0

        print("Processing image dataset...")

        if not os.path.exists(img_path):
            print(f"Error: image directory does not exist - {img_path}")
            return

        subfolders = [f for f in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, f))]
        total_items = len(subfolders)
        print(f"Found {total_items} subdirectories")

        all_valid_items = []

        print("Stage 1: collecting valid items...")
        for name in subfolders:
            if len(all_valid_items) >= limit:
                print(f"Collected {len(all_valid_items)} valid items; reached the collection limit")
                break

            image_folder_path = os.path.join(img_path, name)

            selected_images = get_images_from_folder(image_folder_path, num_images=1)
            if not selected_images:
                skipped_items += 1
                print(f"Skipping item {processed_items + skipped_items}: not enough images - {image_folder_path}")
                continue

            subfolder_path = os.path.join(base_dir, name)
            txt_filename = f"{name}_bytes.txt"
            txt_path = os.path.join(subfolder_path, txt_filename)

            if not os.path.exists(txt_path):
                skipped_items += 1
                print(f"Skipping item {processed_items + skipped_items}: token file does not exist - {txt_path}")
                continue

            token_sequence = read_token_sequence(txt_path)
            if not token_sequence:
                skipped_items += 1
                print(f"Skipping item {processed_items + skipped_items}: token sequence is empty")
                continue

            processed_items += 1

            all_valid_items.append({"name": name, "images": selected_images, "token_sequence": token_sequence})

            if processed_items % 100 == 0:
                print(f"Collected {processed_items} valid items")

        print(f"Collection complete: {len(all_valid_items)} valid items")

        print("Stage 2: splitting training and test data...")

        if test_json_path and os.path.exists(test_json_path):
            # Exclude the supplied test split from training to prevent data leakage.
            print(f"Loading test split: {test_json_path}")
            with open(test_json_path, "r", encoding="utf-8") as f:
                test_json_data = json.load(f)

            test_names = set()
            for item in test_json_data:
                if "name" in item:
                    test_names.add(item["name"])

            print(f"Test split contains {len(test_names)} items")

            # Exclude held-out models before creating training conversations.
            train_items = [item for item in all_valid_items if item["name"] not in test_names]
            test_items = []  # Reuse the supplied test split.

            print(f"Training split: {len(train_items)} items (held-out test items excluded)")
            print(f"Excluded {len(all_valid_items) - len(train_items)} held-out test items")
        else:
            # Create a random split when no existing test split is available.
            if test_json_path:
                print(f"Warning: test split file does not exist - {test_json_path}; using a random split")
            print("Randomly splitting training and test data...")
            random.shuffle(all_valid_items)

            split_index = int(len(all_valid_items) * 0.9)
            train_items = all_valid_items[:split_index]
            test_items = all_valid_items[split_index:]

            print(f"Training split: {len(train_items)} items")
            print(f"Test split: {len(test_items)} items")

        print("Stage 3: generating training conversations...")
        current_data = []
        file_counter = 1
        total_conversations = 0

        def save_current_file():
            """Write the current conversation shard and clear its buffer."""
            nonlocal file_counter, current_data
            if not current_data:
                return

            output_path = f"{output_dir}/mllm_shapenet_dataset_{mesh_type}_{file_counter}_{dataset_type}.json"
            print(f"Saving results to {output_path}...")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(current_data, f, ensure_ascii=False, indent=2)

            file_size = os.path.getsize(output_path)
            print(f"File {output_path} size: {file_size / (1024 * 1024):.2f} MB")

            file_counter += 1
            current_data = []

        def check_file_size_and_save():
            """Flush the current shard when its estimated size approaches the limit."""
            if not current_data:
                return False

            # Estimate the serialized JSON size before writing a shard.
            estimated_size = len(json.dumps(current_data, ensure_ascii=False, indent=2).encode("utf-8"))

            if estimated_size >= max_file_size * 0.9:  # Leave 10% headroom for the final serialized shard.
                save_current_file()
                return True
            return False

        for item in train_items:
            conversation = generate_image_conversation(item["token_sequence"], item["images"])
            current_data.append(conversation)
            total_conversations += 1

            # Check shard size every 100 conversations to limit serialization overhead.
            if total_conversations % 100 == 0:
                check_file_size_and_save()
                print(f"Generated {total_conversations} training conversations")

        if current_data:
            save_current_file()

        print(f"Training data complete: {total_conversations} conversations saved in {file_counter - 1} files")

        # Write a test split only when one was not supplied.
        if test_items:
            print("Stage 4: generating test data...")
            test_data = []

            for item in test_items:
                test_data.append({"name": item["name"], "images": item["images"]})

            test_output_path = f"{output_dir}/mllm_shapenet_test_{mesh_type}_{dataset_type}.json"
            print(f"Saving test split to {test_output_path}...")
            with open(test_output_path, "w", encoding="utf-8") as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)

            test_file_size = os.path.getsize(test_output_path)
            print(f"Test split file size: {test_file_size / (1024 * 1024):.2f} MB")
        else:
            print("Reusing the supplied test split.")

        print("\n=== Processing complete ===")
        print(f"Total items: {total_items}")
        print(f"Processed items: {processed_items}")
        print(f"Skipped items: {skipped_items}")
        print(f"Training conversations: {total_conversations}")
        print(f"Training shards: {file_counter - 1}")
        if test_items:
            print(f"Test items: {len(test_items)}")
        else:
            print("Test split: using the supplied file")
        print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
