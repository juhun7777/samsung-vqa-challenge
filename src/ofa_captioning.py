#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OFA Model Image Captioning System (Stage 1)

This script generates rich image descriptions using multi-question strategy.
Output: CSV file with 'inform' column containing concatenated captions.

Environment:
- Python: 3.9.21
- GPU: NVIDIA GeForce GTX 1080 Ti
- CUDA: 11.7

Author: Juhun Lee
Date: 2025-08-04
"""

import os
import re
import json
import time
import argparse
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
from transformers.models.ofa.generate import sequence_generator
from tqdm import tqdm


def save_ofa_model_weights(model, tokenizer, generator, save_dir="./weights/ofa_model_weights"):
    """Save OFA model, tokenizer, and generator weights"""
    print(f"\n=== Saving OFA Model Weights ===")
    os.makedirs(save_dir, exist_ok=True)

    try:
        # Save OFA model
        ofa_model_path = os.path.join(save_dir, "ofa_model")
        model.save_pretrained(ofa_model_path)
        print(f"[OK] OFA model saved: {ofa_model_path}")

        # Save OFA tokenizer
        ofa_tokenizer_path = os.path.join(save_dir, "ofa_tokenizer")
        tokenizer.save_pretrained(ofa_tokenizer_path)
        print(f"[OK] OFA tokenizer saved: {ofa_tokenizer_path}")

        # Save OFA config
        ofa_config = {
            "model_name": "OFA-base",
            "ofa_model_path": ofa_model_path,
            "ofa_tokenizer_path": ofa_tokenizer_path,
            "image_resolution": 480,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "generator_settings": {
                "beam_size": 8,
                "max_len_b": 120,
                "min_len": 20,
                "no_repeat_ngram_size": 3,
                "temperature": 1.0
            },
            "interpolation": "BICUBIC",
            "save_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "torch_dtype": "float32",
            "device": "cuda",
            "model_type": "ofa_vqa"
        }

        config_path = os.path.join(save_dir, "ofa_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(ofa_config, f, ensure_ascii=False, indent=2)
        print(f"[OK] OFA config saved: {config_path}")

        print(f"\n[SUCCESS] OFA model weights saved!")
        print(f"Location: {save_dir}")

        return save_dir

    except Exception as e:
        print(f"[ERROR] Failed to save OFA model weights: {e}")
        raise e


def load_saved_ofa_model(save_dir="./weights/ofa_model_weights"):
    """Load saved OFA model weights"""
    try:
        print(f"Loading saved OFA model: {save_dir}")

        ofa_model_path = os.path.join(save_dir, "ofa_model")
        ofa_tokenizer_path = os.path.join(save_dir, "ofa_tokenizer")
        config_path = os.path.join(save_dir, "ofa_config.json")

        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Load tokenizer
        tokenizer = OFATokenizer.from_pretrained(ofa_tokenizer_path)
        print("[OK] OFA tokenizer loaded!")

        # Load model
        model = OFAModel.from_pretrained(ofa_model_path, use_cache=True).to("cuda")
        print("[OK] OFA model loaded!")

        # Setup generator
        generator = sequence_generator.SequenceGenerator(
            tokenizer=tokenizer,
            beam_size=config["generator_settings"]["beam_size"],
            max_len_b=config["generator_settings"]["max_len_b"],
            min_len=config["generator_settings"]["min_len"],
            no_repeat_ngram_size=config["generator_settings"]["no_repeat_ngram_size"],
            temperature=config["generator_settings"]["temperature"]
        )
        print("[OK] OFA generator configured!")

        # Image preprocessing
        patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((config["image_resolution"], config["image_resolution"]),
                            interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=config["mean"], std=config["std"])
        ])
        print("[OK] Image preprocessing configured!")

        return model, tokenizer, generator, patch_resize_transform, config

    except Exception as e:
        print(f"[ERROR] Failed to load saved OFA model: {e}")
        return None, None, None, None, None


def clean_caption(caption):
    """Remove bin tokens from OFA caption"""
    cleaned = re.sub(r'<bin_\d+>', '', caption)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()
    return cleaned


def ofa_captioning(model, tokenizer, generator, patch_resize_transform, image_path, prompt):
    """OFA image captioning function"""
    raw_image = Image.open(image_path).convert('RGB')

    txt = f" {prompt}"
    inputs = tokenizer([txt], return_tensors="pt").input_ids.to("cuda")
    patch_img = patch_resize_transform(raw_image).unsqueeze(0).to("cuda")

    data = {
        "net_input": {
            "input_ids": inputs,
            'patch_images': patch_img,
            'patch_masks': torch.tensor([True]).to("cuda")
        }
    }

    with torch.no_grad():
        gen_output = generator.generate([model], data)
        gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]

    caption = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    cleaned_caption = clean_caption(caption)
    return cleaned_caption


def process_single_image(model, tokenizer, generator, patch_resize_transform, image_path, specific_question):
    """Process single image with multi-question strategy"""

    # VQA-optimized questions
    questions = [
        "what do you see in the image?",
        "what does the image describe?",
        "what are the objects in the image?",
        specific_question
    ]

    valid_answers = []

    for question in questions:
        try:
            answer = ofa_captioning(model, tokenizer, generator, patch_resize_transform, image_path, question)

            # Check if valid caption
            if answer.strip().lower() not in ['no', 'yes'] and answer.strip() != "":
                valid_answers.append(answer.strip())

        except Exception as e:
            print(f"Error processing question: {e}")
            continue

    # Concatenate valid captions
    if valid_answers:
        final_caption = ". ".join(valid_answers)
        return final_caption
    else:
        return ""


def main():
    parser = argparse.ArgumentParser(description='OFA Image Captioning')
    parser.add_argument('--model_path', type=str, default='./OFA-base',
                        help='Path to OFA model (or Hugging Face model name)')
    parser.add_argument('--weights_dir', type=str, default='./weights/ofa_model_weights',
                        help='Directory to save/load model weights')
    parser.add_argument('--input_csv', type=str, default='./data/test.csv',
                        help='Input CSV file path')
    parser.add_argument('--output_csv', type=str, default='./data/ofa_caption_test.csv',
                        help='Output CSV file path')
    parser.add_argument('--image_folder', type=str, default='./data/test_input_images',
                        help='Image folder path')
    parser.add_argument('--use_saved', action='store_true',
                        help='Use saved model weights')

    args = parser.parse_args()

    # Load model
    if args.use_saved:
        model, tokenizer, generator, patch_resize_transform, config = load_saved_ofa_model(args.weights_dir)
    else:
        print("Loading OFA model from Hugging Face...")
        tokenizer = OFATokenizer.from_pretrained(args.model_path)
        model = OFAModel.from_pretrained(args.model_path, use_cache=True).to("cuda")

        # Image preprocessing
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 480

        patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        # Generator setup
        generator = sequence_generator.SequenceGenerator(
            tokenizer=tokenizer,
            beam_size=8,
            max_len_b=120,
            min_len=20,
            no_repeat_ngram_size=3,
            temperature=1.0,
        )

        # Save weights
        save_ofa_model_weights(model, tokenizer, generator, args.weights_dir)

    # Load CSV
    df = pd.read_csv(args.input_csv)
    inform_list = []

    # Progress tracking
    start_time = time.time()
    successful_count = 0
    failed_count = 0

    print(f"Processing {len(df)} images...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="OFA Captioning"):
        try:
            image_id = row['ID']

            # Find image file
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                potential_path = os.path.join(args.image_folder, f"{image_id}{ext}")
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break

            if img_path is None:
                print(f"Image not found: {image_id}")
                inform_list.append("")
                failed_count += 1
                continue

            # Get specific question
            specific_question = row['Question']

            # Generate description
            description = process_single_image(model, tokenizer, generator, patch_resize_transform, img_path, specific_question)

            inform_list.append(description)

            if description.strip():
                successful_count += 1
            else:
                failed_count += 1

            # Progress output
            if (idx + 1) % 100 == 0:
                elapsed_time = time.time() - start_time
                avg_time = elapsed_time / (idx + 1)
                remaining_time = avg_time * (len(df) - idx - 1)

                print(f"\n{'='*50}")
                print(f"Progress: {idx + 1}/{len(df)} ({(idx + 1)/len(df)*100:.1f}%)")
                print(f"Success: {successful_count}, Failed: {failed_count}")
                print(f"Est. remaining: {remaining_time/60:.1f} min")
                print("=" * 50)

            # GPU memory cleanup
            if (idx + 1) % 50 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nError processing {image_id}: {e}")
            inform_list.append("")
            failed_count += 1
            continue

    # Add inform column
    df['inform'] = inform_list

    # Save output CSV
    df.to_csv(args.output_csv, index=False, encoding='utf-8-sig')

    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("OFA Captioning Completed!")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.1f} min")
    print(f"Total images: {len(df)}")
    print(f"Success: {successful_count}")
    print(f"Failed: {failed_count}")
    print(f"Success rate: {successful_count/len(df)*100:.1f}%")
    print(f"Output file: {args.output_csv}")

    # Memory cleanup
    print("\nCleaning up memory...")
    del model
    del tokenizer
    torch.cuda.empty_cache()
    print("Memory cleaned")


if __name__ == "__main__":
    main()
