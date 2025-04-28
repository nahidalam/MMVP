import csv
import os
import argparse
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor
import numpy as np

def benchmark_model_siglip2(model_name, benchmark_dir, device="cuda"):
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    image_dir = os.path.join(benchmark_dir, 'MLLM_VLM Images')
    csv_file = os.path.join(benchmark_dir, 'Questions.csv')

    csv_outfile = open('output_siglip2.csv', 'w', newline='')
    csv_writer = csv.writer(csv_outfile)
    csv_writer.writerow(['qid1', 'qid2', 'pred1', 'pred2', 'gt1', 'gt2', 'q1score', 'q2score'])  # header

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
        for i, row in enumerate(reader):
            qid1, qtype1, statement1 = row

            row = next(reader, None)
            if not row:
                break
            qid2, qtype2, statement2 = row

            qid1, qid2 = int(qid1), int(qid2)

            img1 = Image.open(os.path.join(image_dir, qtype1, f'{qid1}.jpg')).convert("RGB")
            img2 = Image.open(os.path.join(image_dir, qtype1, f'{qid2}.jpg')).convert("RGB")

            text1 = f"This is a photo of {statement1}."
            text2 = f"This is a photo of {statement2}."

            # Process images and texts
            imgs = [img1, img2]

            inputs1 = processor(
                text=[text1],
                images=imgs,
                padding="max_length",
                max_length=64,
                return_tensors="pt"
            ).to(device)

            inputs2 = processor(
                text=[text2],
                images=imgs,
                padding="max_length",
                max_length=64,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs1 = model(**inputs1)
                outputs2 = model(**inputs2)

            logits_per_image1 = outputs1.logits_per_image  # shape: (2, 1)
            logits_per_image2 = outputs2.logits_per_image  # shape: (2, 1)

            probs1 = torch.sigmoid(logits_per_image1).squeeze().cpu().numpy()  # shape: (2,)
            probs2 = torch.sigmoid(logits_per_image2).squeeze().cpu().numpy()  # shape: (2,)

            img1_score1 = probs1[0]
            img1_score2 = probs2[0]

            pred1 = "img1" if img1_score1 > img1_score2 else "img2"
            pred2 = "img1" if img1_score2 > img1_score1 else "img2"

            gt1 = "img1" if qid1 % 2 == 1 else "img2"
            gt2 = "img1" if qid2 % 2 == 1 else "img2"

            csv_writer.writerow([qid1, qid2, pred1, pred2, gt1, gt2, img1_score1, img1_score2])

            current_category = categories[num_pairs // 15]
            if pred1 == gt1 and pred2 == gt2:
                pair_accuracies[current_category] += 1
            num_pairs += 1

    csv_outfile.close()

    # Calculate percentage accuracies
    for category in pair_accuracies:
        pair_accuracies[category] = (pair_accuracies[category] / (num_pairs // len(categories))) * 100

    return pair_accuracies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark SigLIP2 model.')
    parser.add_argument('--directory', type=str, required=True, help='The path to the benchmark directory')
    parser.add_argument('--model_name', type=str, default='google/siglip2-base-patch16-224', help='HuggingFace model name')
    args = parser.parse_args()
    model_name = args.model_name
    results_siglip2 = {model_name: benchmark_model_siglip2(args.model_name, args.directory)}

    categories = results_siglip2[model_name].keys()
    data = {'Categories': list(categories)}
    for model in list(results_siglip2.keys()):
        data[model] = [results_siglip2[model][category] for category in categories]

    print(results_siglip2)

