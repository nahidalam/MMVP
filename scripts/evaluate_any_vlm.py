import os
import csv
import argparse
from PIL import Image

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoProcessor, AutoModel, CLIPModel, CLIPProcessor
is_openai_clip = None

def load_model(model_name, device):
    """
    Loads either OpenAI CLIP or HuggingFace-compatible model.
    """
    if model_name.startswith("ViT") or model_name.lower().startswith("openai"):
        import clip
        is_openai_clip = True
        model, preprocess = clip.load(model_name, device=device)
        return model, preprocess
    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor

def benchmark_model(model_name, benchmark_dir, device="cpu"):
    model, preprocess = load_model(model_name, device)
    #is_openai_clip = tokenizer is not None

    image_dir = os.path.join(benchmark_dir, 'MLLM_VLM Images')
    csv_file = os.path.join(benchmark_dir, 'Questions.csv')
    csv_outfile = open('output.csv', 'w', newline='')
    csv_writer = csv.writer(csv_outfile)
    csv_writer.writerow(['qid1', 'qid2', 'pred1', 'pred2', 'gt1', 'gt2', 'q1score', 'q2score'])

    categories = [
        'Orientation and Direction', 'Presence of Specific Features', 
        'State and Condition', 'Quantity and Count', 
        'Positional and Relational Context', 'Color and Appearance',
        'Structural Characteristics', 'Texts',
        'Viewpoint and Perspective'
    ]

    pair_accuracies = {category: 0 for category in categories}
    num_pairs = 0

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            qid1, qtype1, statement1 = row
            row = next(reader, None)
            if not row:
                break
            qid2, qtype2, statement2 = row

            qid1, qid2 = int(qid1), int(qid2)

            img1_path = os.path.join(image_dir, qtype1, f'{qid1}.jpg')
            img2_path = os.path.join(image_dir, qtype1, f'{qid2}.jpg')
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")

            text1 = 'a photo of ' + statement1
            text2 = 'a photo of ' + statement2

            if is_openai_clip:
                text1_token = tokenizer([text1]).to(device)
                text2_token = tokenizer([text2]).to(device)

                img1_tensor = preprocess(img1).unsqueeze(0).to(device)
                img2_tensor = preprocess(img2).unsqueeze(0).to(device)
                imgs = torch.cat((img1_tensor, img2_tensor), dim=0)

                with torch.no_grad():
                    logits_per_image1, logits_per_text1 = model(imgs, text1_token)
                    logits_per_image2, logits_per_text2 = model(imgs, text2_token)

                    probs1 = logits_per_text1.softmax(dim=-1).cpu().numpy()
                    probs2 = logits_per_text2.softmax(dim=-1).cpu().numpy()
            else:
                if "siglip2" in model_name.lower():
                    inputs1 = preprocess(
                        text=[text1] * 2,
                        images=[img1, img2],
                        padding="max_length",
                        max_length=64,
                        return_tensors="pt"
                    ).to(device)

                    inputs2 = preprocess(
                        text=[text2] * 2,
                        images=[img1, img2],
                        padding="max_length",
                        max_length=64,
                        return_tensors="pt"
                    ).to(device)
                else:
                    inputs1 = preprocess(
                        text=[text1] * 2,
                        images=[img1, img2],
                        return_tensors="pt",
                        padding=True
                    ).to(device)

                    inputs2 = preprocess(
                        text=[text2] * 2,
                        images=[img1, img2],
                        return_tensors="pt",
                        padding=True
                    ).to(device)

                with torch.no_grad():
                    outputs1 = model(**inputs1)
                    outputs2 = model(**inputs2)

                probs1 = outputs1.logits_per_image.softmax(dim=-1).cpu().numpy()
                probs2 = outputs2.logits_per_image.softmax(dim=-1).cpu().numpy()

            img1_score1 = probs1[0][0]
            img1_score2 = probs2[0][0]

            pred1 = "img1" if img1_score1 > 0.5 else "img2"
            pred2 = "img1" if img1_score2 > 0.5 else "img2"

            gt1 = "img1" if qid1 % 2 == 1 else "img2"
            gt2 = "img1" if qid2 % 2 == 1 else "img2"

            csv_writer.writerow([qid1, qid2, pred1, pred2, gt1, gt2, img1_score1, img1_score2])

            current_category = categories[num_pairs // 15]
            if pred1 == gt1 and pred2 == gt2:
                pair_accuracies[current_category] += 1
            num_pairs += 1

    csv_outfile.close()

    for category in pair_accuracies:
        pair_accuracies[category] = (pair_accuracies[category] / (num_pairs // len(categories))) * 100

    return pair_accuracies

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark vision-language model.')
    parser.add_argument('--directory', type=str, required=True, help='Benchmark directory path')
    parser.add_argument('--model_name', type=str, required=True, help='HuggingFace or OpenAI CLIP model name')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    args = parser.parse_args()

    results = benchmark_model(args.model_name, args.directory, args.device)
    print(results)

