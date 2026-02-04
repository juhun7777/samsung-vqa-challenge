#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VQA (Visual Question Answering) Inference System
OFA model for image captioning + Phi-2 model for answer selection

Environment:
- OS: Windows 11
- Python: 3.8+
- PyTorch: 1.13.0+
- Transformers: 4.21.0+
- CUDA: 11.8+

Author: Juhun Lee
Date: 2025-08-04
Competition: Samsung Collegiate Programming Challenge 2025
"""

import os
import json
import time
import argparse
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from transformers import (
    OFATokenizer, OFAModel,
    AutoTokenizer, AutoModelForCausalLM
)
from transformers.models.ofa.generate import sequence_generator
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


class VQAInferenceSystem:
    """OFA + Phi-2 based VQA Inference System"""

    def __init__(self, ofa_weights_dir="./weights/ofa_model_weights",
                 phi_weights_dir="./weights/phi_model_weights", device="cuda"):
        """
        Args:
            ofa_weights_dir (str): OFA model weights directory
            phi_weights_dir (str): Phi-2 model weights directory
            device (str): Execution device ('cuda' or 'cpu')
        """
        self.device = device
        self.ofa_weights_dir = ofa_weights_dir
        self.phi_weights_dir = phi_weights_dir

        # Model initialization
        self.ofa_model = None
        self.ofa_tokenizer = None
        self.ofa_generator = None
        self.ofa_transform = None
        self.ofa_config = None

        self.phi_model = None
        self.phi_tokenizer = None

        print(f"VQA Inference System Initialization...")
        print(f"Device: {self.device}")
        print(f"OFA weights path: {self.ofa_weights_dir}")
        print(f"Phi weights path: {self.phi_weights_dir}")

    def load_ofa_model(self):
        """Load OFA model weights"""
        try:
            print("\n=== Loading OFA Model ===")

            # Load config
            config_path = os.path.join(self.ofa_weights_dir, "ofa_config.json")
            with open(config_path, 'r', encoding='utf-8') as f:
                self.ofa_config = json.load(f)

            # Model and tokenizer paths
            ofa_model_path = os.path.join(self.ofa_weights_dir, "ofa_model")
            ofa_tokenizer_path = os.path.join(self.ofa_weights_dir, "ofa_tokenizer")

            # Load tokenizer
            self.ofa_tokenizer = OFATokenizer.from_pretrained(ofa_tokenizer_path)
            print("[OK] OFA Tokenizer loaded")

            # Load model
            self.ofa_model = OFAModel.from_pretrained(
                ofa_model_path,
                use_cache=True
            ).to(self.device)
            self.ofa_model.eval()
            print("[OK] OFA Model loaded")

            # Generator settings
            generator_settings = self.ofa_config["generator_settings"]
            self.ofa_generator = sequence_generator.SequenceGenerator(
                tokenizer=self.ofa_tokenizer,
                beam_size=generator_settings["beam_size"],
                max_len_b=generator_settings["max_len_b"],
                min_len=generator_settings["min_len"],
                no_repeat_ngram_size=generator_settings["no_repeat_ngram_size"],
                temperature=generator_settings["temperature"]
            )
            print("[OK] OFA Generator configured")

            # Image preprocessing
            self.ofa_transform = transforms.Compose([
                lambda image: image.convert("RGB"),
                transforms.Resize(
                    (self.ofa_config["image_resolution"],
                     self.ofa_config["image_resolution"]),
                    interpolation=Image.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.ofa_config["mean"],
                    std=self.ofa_config["std"]
                )
            ])
            print("[OK] OFA Image preprocessing configured")

            return True

        except Exception as e:
            print(f"[ERROR] OFA Model loading failed: {e}")
            return False

    def load_phi_model(self):
        """Load Phi-2 model weights"""
        try:
            print("\n=== Loading Phi-2 Model ===")

            # Model and tokenizer paths
            phi_model_path = os.path.join(self.phi_weights_dir, "phi_model")
            phi_tokenizer_path = os.path.join(self.phi_weights_dir, "phi_tokenizer")

            # Load tokenizer
            self.phi_tokenizer = AutoTokenizer.from_pretrained(
                phi_tokenizer_path,
                trust_remote_code=True
            )

            if self.phi_tokenizer.pad_token is None:
                self.phi_tokenizer.pad_token = self.phi_tokenizer.eos_token

            print("[OK] Phi-2 Tokenizer loaded")

            # Load model
            self.phi_model = AutoModelForCausalLM.from_pretrained(
                phi_model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).to(self.device)

            self.phi_model.eval()
            print("[OK] Phi-2 Model loaded")

            return True

        except Exception as e:
            print(f"[ERROR] Phi-2 Model loading failed: {e}")
            return False

    def clean_caption(self, caption):
        """Remove bin tokens from OFA caption"""
        import re
        cleaned = re.sub(r'<bin_\d+>', '', caption)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        return cleaned

    def generate_image_caption(self, image_path, question):
        """Generate image caption using OFA model"""
        try:
            # Load image
            raw_image = Image.open(image_path).convert('RGB')

            # Multi-question strategy for VQA optimization
            questions = [
                "what do you see in the image?",
                "what does the image describe?",
                "what are the objects in the image?",
                question if question and str(question).strip() not in ['', 'nan', 'None'] else "what is in this image?"
            ]

            valid_captions = []

            for q in questions:
                try:
                    # Prepare prompt
                    txt = f" {q}"
                    inputs = self.ofa_tokenizer([txt], return_tensors="pt").input_ids.to(self.device)
                    patch_img = self.ofa_transform(raw_image).unsqueeze(0).to(self.device)

                    # OFA input data
                    data = {
                        "net_input": {
                            "input_ids": inputs,
                            'patch_images': patch_img,
                            'patch_masks': torch.tensor([True]).to(self.device)
                        }
                    }

                    # Inference
                    with torch.no_grad():
                        gen_output = self.ofa_generator.generate([self.ofa_model], data)
                        gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]

                    # Decode and clean caption
                    caption = self.ofa_tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
                    cleaned_caption = self.clean_caption(caption)

                    # Check if valid caption
                    if cleaned_caption.strip().lower() not in ['no', 'yes'] and cleaned_caption.strip():
                        valid_captions.append(cleaned_caption.strip())

                except Exception as e:
                    print(f"Error processing question: {e}")
                    continue

            # Concatenate all captions
            if valid_captions:
                final_caption = ". ".join(valid_captions)
                return final_caption
            else:
                return "Unable to generate image description"

        except Exception as e:
            print(f"Error generating image caption: {e}")
            return "Error in image captioning"

    def create_phi_prompt(self, image_description, question, choices):
        """Create prompt for Phi-2 model"""
        if question and str(question).strip() not in ['', 'nan', 'None']:
            prompt = f"""Based on the image description and question, select the most probable answer

Image: {image_description}
Question: {question}
A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}

Answer:"""
        else:
            prompt = f"""Based on the image description, select the most appropriate answer.

Image: {image_description}
A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}

Answer:"""
        return prompt

    def extract_answer_from_response(self, response):
        """Extract answer from Phi-2 response"""
        try:
            response = response.strip().upper()

            # Pattern matching
            if 'A)' in response or response.startswith('A'):
                return 'A'
            elif 'B)' in response or response.startswith('B'):
                return 'B'
            elif 'C)' in response or response.startswith('C'):
                return 'C'
            elif 'D)' in response or response.startswith('D'):
                return 'D'

            # Find A, B, C, D characters
            for char in response:
                if char in ['A', 'B', 'C', 'D']:
                    return char

            # Default value
            return 'A'

        except Exception:
            return 'A'

    def predict_answer(self, image_description, question, choices):
        """Predict answer using Phi-2 model"""
        try:
            # Create prompt
            prompt = self.create_phi_prompt(image_description, question, choices)

            # Tokenize
            inputs = self.phi_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=400,
                padding=False,
                add_special_tokens=True
            )

            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # Model inference
            with torch.no_grad():
                outputs = self.phi_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=15,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.phi_tokenizer.eos_token_id,
                    eos_token_id=self.phi_tokenizer.eos_token_id,
                    temperature=0.0,
                    use_cache=True
                )

            # Decode response
            new_tokens = outputs[0][input_ids.shape[1]:]
            response = self.phi_tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Extract answer
            answer = self.extract_answer_from_response(response)

            return answer

        except Exception as e:
            print(f"Error predicting answer: {e}")
            return 'A'  # Default fallback

    def run_inference(self, test_csv_path="./data/test.csv",
                     test_images_dir="./data/test_input_images",
                     output_path="./results/submission.csv"):
        """Run inference on entire test dataset"""
        print(f"\n{'='*80}")
        print("VQA Inference Started")
        print(f"{'='*80}")

        # Load models
        if not self.load_ofa_model():
            print("[ERROR] OFA model loading failed")
            return False

        if not self.load_phi_model():
            print("[ERROR] Phi-2 model loading failed")
            return False

        print(f"\n[OK] All models loaded successfully!")

        # Load test data
        print(f"\nLoading test data: {test_csv_path}")
        df = pd.read_csv(test_csv_path)
        print(f"Total {len(df)} test samples")

        # Results list
        results = []
        start_time = time.time()

        print(f"\nStarting inference...")

        # Process each test sample
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="VQA Inference"):
            try:
                # Extract data
                image_id = str(row['ID'])
                question = str(row.get('Question', '')) if 'Question' in row else None
                choices = [str(row['A']), str(row['B']), str(row['C']), str(row['D'])]

                # Image path (try multiple extensions)
                image_path = None
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    potential_path = os.path.join(test_images_dir, f"{image_id}{ext}")
                    if os.path.exists(potential_path):
                        image_path = potential_path
                        break

                if image_path is None:
                    print(f"Image file not found: {image_id}")
                    results.append({'ID': image_id, 'answer': 'A'})
                    continue

                # Stage 1: OFA image captioning
                image_description = self.generate_image_caption(image_path, question)

                # Stage 2: Phi-2 answer prediction
                predicted_answer = self.predict_answer(image_description, question, choices)

                # Save result
                results.append({
                    'ID': image_id,
                    'answer': predicted_answer
                })

                # Periodic GPU memory cleanup
                if idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Progress output (every 100 samples)
                if (idx + 1) % 100 == 0:
                    elapsed_time = time.time() - start_time
                    avg_time = elapsed_time / (idx + 1)
                    remaining_time = avg_time * (len(df) - idx - 1)

                    print(f"\nProgress: {idx + 1}/{len(df)} ({(idx + 1)/len(df)*100:.1f}%)")
                    print(f"Estimated remaining time: {remaining_time/60:.1f} min")

            except Exception as e:
                print(f"Error processing sample (ID: {row.get('ID', 'Unknown')}): {e}")
                results.append({
                    'ID': str(row.get('ID', f'ERROR_{idx}')),
                    'answer': 'A'
                })

        # Save results
        print(f"\nSaving results...")

        # Create results directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create result DataFrame and save
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        # Processing summary
        total_time = time.time() - start_time

        print(f"\n{'='*80}")
        print("VQA Inference Completed!")
        print(f"{'='*80}")
        print(f"Total processing time: {total_time/60:.1f} min")
        print(f"Total processed samples: {len(results)}")
        print(f"Output file: {output_path}")

        # Answer distribution
        answer_counts = result_df['answer'].value_counts()
        print(f"\nAnswer distribution:")
        for answer, count in sorted(answer_counts.items()):
            print(f"  {answer}: {count} ({count/len(result_df)*100:.1f}%)")

        # Save summary
        summary_info = {
            'total_samples': len(df),
            'processed_samples': len(results),
            'processing_time_minutes': total_time / 60,
            'models_used': ['OFA-base', 'microsoft/phi-2'],
            'answer_distribution': dict(answer_counts),
            'inference_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'random_seed': RANDOM_SEED,
            'device': self.device
        }

        summary_path = output_path.replace('.csv', '_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_info, f, ensure_ascii=False, indent=2)

        print(f"Summary file: {summary_path}")

        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"\nGPU memory cleaned")

        return True


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='VQA Inference System')
    parser.add_argument('--test_csv', type=str, default='./data/test.csv',
                        help='Path to test CSV file')
    parser.add_argument('--test_images', type=str, default='./data/test_input_images',
                        help='Path to test images directory')
    parser.add_argument('--output', type=str, default='./results/submission.csv',
                        help='Output submission file path')
    parser.add_argument('--ofa_weights', type=str, default='./weights/ofa_model_weights',
                        help='OFA model weights directory')
    parser.add_argument('--phi_weights', type=str, default='./weights/phi_model_weights',
                        help='Phi-2 model weights directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    print("VQA Inference System")
    print(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        device = args.device
    else:
        print("CUDA not available, using CPU")
        device = "cpu"

    # Initialize VQA system
    vqa_system = VQAInferenceSystem(
        ofa_weights_dir=args.ofa_weights,
        phi_weights_dir=args.phi_weights,
        device=device
    )

    # Run inference
    success = vqa_system.run_inference(
        test_csv_path=args.test_csv,
        test_images_dir=args.test_images,
        output_path=args.output
    )

    if success:
        print("\n[SUCCESS] VQA inference completed successfully!")
    else:
        print("\n[ERROR] VQA inference failed.")

    return success


if __name__ == "__main__":
    main()
