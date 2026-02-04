#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi-2 Model VQA Reasoning System (Stage 2)

This script takes OFA-generated captions and predicts answers
using Phi-2's reasoning capabilities.

Environment:
- Python: 3.10.18
- GPU: NVIDIA GeForce GTX 1080 Ti
- CUDA: 11.7

Author: Juhun Lee
Date: 2025-08-04
"""

import os
import json
import time
import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Matplotlib font settings (for Korean)
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# torch.compiler error workaround
if not hasattr(torch, 'compiler'):
    torch.compiler = None


class PhiVQASystem:
    def __init__(self, device="cuda"):
        """
        Phi-2 based VQA System
        """
        self.device = device
        self.phi_model = None
        self.phi_tokenizer = None

    def load_phi_model(self, model_name="microsoft/phi-2"):
        """Load Phi model from Hugging Face"""
        try:
            print("Loading Phi-2 model...")

            self.phi_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            if self.phi_tokenizer.pad_token is None:
                self.phi_tokenizer.pad_token = self.phi_tokenizer.eos_token

            self.phi_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).to(self.device)

            self.phi_model.eval()
            print("[OK] Phi-2 model loaded!")

        except Exception as e:
            print(f"[ERROR] Phi model loading failed: {e}")
            raise e

    def save_model_weights(self, weights_dir="./weights/phi_model_weights"):
        """Save model weights"""
        print(f"\n=== Saving Phi Model Weights ===")
        os.makedirs(weights_dir, exist_ok=True)

        try:
            phi_model_path = os.path.join(weights_dir, "phi_model")
            phi_tokenizer_path = os.path.join(weights_dir, "phi_tokenizer")

            self.phi_model.save_pretrained(phi_model_path)
            self.phi_tokenizer.save_pretrained(phi_tokenizer_path)
            print(f"[OK] Phi model saved: {phi_model_path}")
            print(f"[OK] Phi tokenizer saved: {phi_tokenizer_path}")

            # Save config
            config_info = {
                "device": self.device,
                "model_name": "microsoft/phi-2",
                "phi_model_path": phi_model_path,
                "phi_tokenizer_path": phi_tokenizer_path,
                "save_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "torch_dtype": "float16",
                "model_type": "phi_only_vqa"
            }

            config_path = os.path.join(weights_dir, "model_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_info, f, ensure_ascii=False, indent=2)
            print(f"[OK] Config saved: {config_path}")

            print(f"\n[SUCCESS] Phi model weights saved!")
            print(f"Location: {weights_dir}")

            return weights_dir

        except Exception as e:
            print(f"[ERROR] Failed to save model weights: {e}")
            raise e

    def load_saved_model(self, weights_dir="./weights/phi_model_weights"):
        """Load saved model weights"""
        try:
            print(f"Loading saved model: {weights_dir}")

            phi_model_path = os.path.join(weights_dir, "phi_model")
            phi_tokenizer_path = os.path.join(weights_dir, "phi_tokenizer")

            self.phi_tokenizer = AutoTokenizer.from_pretrained(
                phi_tokenizer_path,
                trust_remote_code=True
            )

            if self.phi_tokenizer.pad_token is None:
                self.phi_tokenizer.pad_token = self.phi_tokenizer.eos_token

            self.phi_model = AutoModelForCausalLM.from_pretrained(
                phi_model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).to(self.device)

            self.phi_model.eval()
            print("[OK] Saved Phi model loaded!")

            return True

        except Exception as e:
            print(f"[ERROR] Failed to load saved model: {e}")
            return False

    def create_phi_prompt(self, image_description, question, choices):
        """Create prompt for Phi model"""
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

    def extract_phi_answer(self, response):
        """Extract answer from Phi model response"""
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

            # Default
            return 'A'

        except Exception:
            return 'A'

    def predict_with_phi(self, image_description, question, choices):
        """Predict answer with Phi model"""
        try:
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
            answer = self.extract_phi_answer(response)

            return answer, response.strip()

        except Exception as e:
            print(f"Phi prediction error: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 'A', str(e)

    def process_dataset(self, csv_path, output_path):
        """Process entire dataset"""
        print(f"Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded: {len(df)} samples")

        print(f"\n=== Processing with Phi Model ===")

        all_results = []
        start_time = time.time()

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Phi Reasoning"):
            try:
                image_id = str(row['ID'])
                image_description = str(row['inform'])
                question = str(row.get('Question', '')) if 'Question' in row else None
                choices = [str(row['A']), str(row['B']), str(row['C']), str(row['D'])]

                # Predict with Phi
                predicted_answer, phi_response = self.predict_with_phi(image_description, question, choices)

                all_results.append({
                    'ID': image_id,
                    'answer': predicted_answer
                })

                # Periodic GPU memory cleanup
                if idx % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error (ID: {row.get('ID', 'Unknown')}): {e}")
                all_results.append({
                    'ID': str(row.get('ID', f'ERROR_{idx}')),
                    'answer': 'A'
                })

        print(f"\nPhi processing completed: {len(all_results)} samples")

        # Save results
        print(f"\n=== Saving Results ===")

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save DataFrame
        final_df = pd.DataFrame(all_results)
        final_df.to_csv(output_path, index=False)
        print(f"[OK] Results saved: {output_path}")

        # Save summary
        summary_info = {
            'total_images': len(df),
            'processed_images': len(all_results),
            'processing_time_minutes': (time.time() - start_time) / 60,
            'model_used': 'phi-2_only',
            'success_rate': len([r for r in all_results if r['answer'] != 'A' or 'ERROR' not in r['ID']]) / len(all_results) * 100
        }

        summary_file = output_path.replace('.csv', '_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_info, f, ensure_ascii=False, indent=2)
        print(f"[OK] Summary saved: {summary_file}")

        # Statistics
        print(f"\n{'='*60}")
        print("Final Results Summary")
        print(f"{'='*60}")
        print(f"Total samples: {len(df)}")
        print(f"Phi processed: {len(all_results)} (100%)")
        print(f"Processing time: {(time.time() - start_time)/60:.1f} min")

        # Answer distribution
        answer_counts = final_df['answer'].value_counts()
        print(f"\nAnswer distribution:")
        for answer, count in sorted(answer_counts.items()):
            print(f"  {answer}: {count} ({count/len(final_df)*100:.1f}%)")

        return final_df, summary_info


def main():
    parser = argparse.ArgumentParser(description='Phi-2 VQA Reasoning')
    parser.add_argument('--input_csv', type=str, default='./data/ofa_caption_test.csv',
                        help='Input CSV file with OFA captions')
    parser.add_argument('--output_csv', type=str, default='./results/submission.csv',
                        help='Output submission file')
    parser.add_argument('--weights_dir', type=str, default='./weights/phi_model_weights',
                        help='Phi model weights directory')
    parser.add_argument('--use_saved', action='store_true',
                        help='Use saved model weights')
    parser.add_argument('--save_weights', action='store_true',
                        help='Save model weights after loading')

    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("CUDA not available, using CPU")
        device = "cpu"

    # Initialize VQA system
    print("Initializing Phi VQA System...")
    vqa_system = PhiVQASystem(device=device)

    # Load model
    if args.use_saved:
        vqa_system.load_saved_model(args.weights_dir)
    else:
        vqa_system.load_phi_model()
        if args.save_weights:
            vqa_system.save_model_weights(args.weights_dir)

    # Process dataset
    print(f"\nStarting dataset processing...")
    result_df, summary_info = vqa_system.process_dataset(args.input_csv, args.output_csv)

    if result_df is not None:
        print(f"\n[SUCCESS] Processing completed!")
        print(f"Results: {args.output_csv}")
        print(f"Summary: {args.output_csv.replace('.csv', '_summary.json')}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"\nGPU memory cleaned")
    else:
        print("[ERROR] Processing failed")


if __name__ == "__main__":
    main()
