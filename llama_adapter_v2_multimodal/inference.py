"""
CUDA_VISIBLE_DEVICES=? python inference.py
"""

import os
import urllib.request
from urllib.parse import urlparse
import csv
import argparse
import cv2
import llama
import torch
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Demo")
parser.add_argument('--llama_dir', type=str, default="../../MiniGPT-4/vicuna_prep/llama_ckpt/")
parser.add_argument('--input_csv', type=str, default='../../MiniGPT-4/input_csv/visit_instructions_700.csv')
parser.add_argument('--output_dir', type=str, default='../../MiniGPT-4/output_csv/')
parser.add_argument('--model_name', type=str, default='LlamaAdapter-v2')
parser.add_argument('--verbose', action='store_true', default=False)
args = parser.parse_args()


def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    return csv_reader.fieldnames, data


def add_backslash_to_spaces(url):
    if ' ' in url:
        url = url.replace(' ', "%20")
    return url


def download_image(url, file_path):
    if args.verbose:
        print(url)
        print(file_path)
    try:
        urllib.request.urlretrieve(url, file_path)
        if args.verbose:
            print("Image downloaded successfully!")
    except urllib.error.URLError as e:
        print("Error occurred while downloading the image:", e)


if __name__ == '__main__':
    # check output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_csv = os.path.join(args.output_dir, f'{args.model_name.lower()}.csv')

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = llama.load("BIAS-7B", args.llama_dir, device)

    # Read CSV file
    fieldname_list, input_data_list = read_csv_file(args.input_csv)

    output_data_list = []
    prediction_fieldname = f'{args.model_name} prediction'
    fieldname_list.append(prediction_fieldname)

    for row in tqdm(input_data_list, total=len(input_data_list), desc='predict'):
        if args.verbose:
            print(row)

        if 'Input.image_url' in row.keys():
            image_url_list = [row['Input.image_url']]
        elif 'image' in row.keys():
            image_url_list = [row['image']]
        else:
            image_url_list = list(eval(row['images']))

        if len(image_url_list) > 1:
            llm_prediction = '[SKIPPED]'

        else:
            # prepare instruction prompt
            prompt = llama.format_prompt(row['instruction'])

            # download image image
            image_url = add_backslash_to_spaces(image_url_list[0])
            extension = image_url.split('.')[-1]
            img_path = os.path.join(os.getcwd(), f'tmp.{extension}')  # Create the local image file path
            download_image(image_url, img_path)

            # load image
            img = Image.fromarray(cv2.imread(img_path))
            img = preprocess(img).unsqueeze(0).to(device)

            # predict
            llm_prediction = model.generate(img, [prompt])[0]
            
            if args.verbose:
                print(f'Question:\n\t{row["instruction"]}')
                print(f'Image URL:\t{image_url}')
                print(f'Answer:\n\t{llm_prediction}')
                print('-'*30 + '\n')

            # Clean up
            os.remove(img_path)

        row[prediction_fieldname] = llm_prediction
        output_data_list.append(row)

    # Write to output csv file
    output_file = args.output_csv
    with open(output_file, 'w', newline='') as file:
        csv_writer = csv.DictWriter(file, fieldnames=fieldname_list)
        csv_writer.writeheader()
        csv_writer.writerows(output_data_list)
