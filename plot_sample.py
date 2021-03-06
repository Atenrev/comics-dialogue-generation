#import libraries
import argparse
import importlib
import torch
import cv2
import os
import pandas as pd
import h5py as h5
import evaluate

from matplotlib import pyplot as plt

from src.common.configuration import get_dataset_configuration, get_model_configuration
from src.datasets.comics_dialogue_generation import ComicsDialogueGenerationDataset
from src.datasets.text_cloze_image_text import ComicsImageTextDataset
from src.datasets.text_cloze_image_text_vlt5 import TextClozeImageTextVLT5Dataset
from src.models.dialogue_generation_vlt5 import DialogueGenerationVLT5Model
from src.tokenizers.vlt5_tokenizers import VLT5TokenizerFast


CORRECT_COLOR = (0.0, 1.0, 0.0)
INCORRECT_COLOR = (1.0, 0.0, 0.0)
GENERATED_COLOR = (0.0, 0.0, 0.0)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Plotting script')

    parser.add_argument('--model', type=str, default="text_cloze_image_text_vlt5",
                        help='Model to run')
    parser.add_argument('--load_cloze_checkpoint', type=str, default="runs/24-TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_hard/models/epoch_10.pt",
                        help='Path to text cloze model checkpoint')
    parser.add_argument('--load_dialogue_checkpoint', type=str, default="runs/DialogueGenerationVLT5Model_comics_dialogue_generation_2022-06-03_00:19:53/models/epoch_10.pt",
                        help='Path to dialogue model checkpoint')
    parser.add_argument('--dataset_config', type=str, default="text_cloze_image_text_vlt5_easy",
                        help='Dataset config to use')
    parser.add_argument('--dataset_dir', type=str, default="datasets/COMICS/",
                        help='Dataset directory path')
    parser.add_argument('--output_dir', type=str, default="plots/",
                        help='Output directory path')
    parser.add_argument('--sample_id', type=int, default=23,
                        help='Sample id to plot')
    parser.add_argument('--seed', type=int, default=4,
                        help='Random seed')

    return parser.parse_args()


def load_datasets(dataset_text_cloze_config, dataset_dir, tokenizer, device):
    df = pd.read_csv(
        f"{dataset_dir}/text_cloze_test_{dataset_text_cloze_config.mode}.csv", delimiter=',')
    df = df.fillna("")

    feats_h5_path = os.path.join(
        dataset_dir, dataset_text_cloze_config.panel_features_path)
    feats_h5 = h5.File(feats_h5_path, 'r')

    dataset_text_cloze = TextClozeImageTextVLT5Dataset(
        df, feats_h5, tokenizer, device, dataset_text_cloze_config)
    return df, dataset_text_cloze


def load_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_checkpoint(checkpoint["model_state_dict"])
    model.eval()
    return model


def main(args) -> None:
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load models
    model_text_cloze_config = get_model_configuration(args.model)
    ModelClass = getattr(importlib.import_module(
        f"src.models.{args.model}"), model_text_cloze_config.classname)
    model_text_cloze = ModelClass(model_text_cloze_config, device).to(device)

    model_dialogue_config = get_model_configuration("dialogue_generation_vlt5")
    model_dialogue = DialogueGenerationVLT5Model(
        model_dialogue_config, device).to(device)

    load_checkpoint(args.load_cloze_checkpoint, model_text_cloze)
    load_checkpoint(args.load_dialogue_checkpoint, model_dialogue)

    tokenizer = VLT5TokenizerFast.from_pretrained(
        model_text_cloze_config.backbone,
        max_length=model_text_cloze_config.max_text_length,
        do_lower_case=model_text_cloze_config.do_lower_case,
    )
    # model_text_cloze.tokenizer = tokenizer
    model_dialogue.tokenizer = tokenizer

    # Load dataset
    dataset_config = get_dataset_configuration(args.dataset_config)
    df, dataset, = load_datasets(
        dataset_config, args.dataset_dir, tokenizer, device)

    sample = df.iloc[args.sample_id]
    book_id = sample["book_id"]
    page_id = sample["page_id"]
    target_text = sample[f"answer_candidate_{sample['correct_answer']}_text"]

    # Run models
    sample_data = dataset[args.sample_id]
    for key in sample_data.keys():
        if isinstance(sample_data[key], torch.Tensor):
            sample_data[key] = sample_data[key].unsqueeze(0).to(device)

    input_ids = sample_data['input_ids'].to(device)
    B = len(input_ids)
    V_L = sample_data['vis_feats'].size(2)
    vis_feats = sample_data['vis_feats'].to(device).view(B, 4*V_L, 2048)
    vis_pos = sample_data['boxes'].to(device).view(B, 4*V_L, 4)

    img_order_ids = [0] * V_L + [1] * V_L + [2] * V_L + [3] * V_L
    img_order_ids = torch.tensor(
        img_order_ids, dtype=torch.long, device=device)
    img_order_ids = img_order_ids.view(1, 4*V_L).expand(B, -1)

    obj_order_ids = torch.arange(V_L, dtype=torch.long, device=device)
    obj_order_ids = obj_order_ids.view(1, 1, V_L).expand(
        B, 4, -1).contiguous().view(B, 4*V_L)

    prediction_text_cloze = model_text_cloze.generate(
        input_ids=input_ids,
        vis_inputs=(vis_feats, vis_pos, img_order_ids, obj_order_ids),
    )
    prediction_text_cloze = tokenizer.batch_decode(
        prediction_text_cloze, skip_special_tokens=True)[0]
    prediction_text_cloze = int(prediction_text_cloze) if prediction_text_cloze.isdigit() else 0

    prediction_dialogue = model_dialogue.generate(
        input_ids=input_ids,
        vis_inputs=(vis_feats, vis_pos, img_order_ids, obj_order_ids),
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    prediction_dialogue = tokenizer.batch_decode(
        prediction_dialogue, skip_special_tokens=True)[0]

    # Calculate metrics
    m_bleu = evaluate.load("google_bleu")
    bleu_score = m_bleu.compute(predictions=[prediction_dialogue], references=[target_text])
    bleu_score = bleu_score["google_bleu"] * 100
    m_meteor = evaluate.load("meteor")
    meteor_score = m_meteor.compute(predictions=[prediction_dialogue], references=[target_text])
    meteor_score = meteor_score["meteor"] * 100

    # Plot sample
    fig = plt.figure(figsize=(16, 8))

    # setting values to rows and column variables
    rows = 2
    columns = 4

    # reading images
    Image1 = cv2.imread(
        f'{args.dataset_dir}/raw_panel_images/{book_id}/{page_id}_{sample["context_panel_0_id"]}.jpg')
    Image1 = cv2.cvtColor(Image1, cv2.COLOR_BGR2RGB)
    Image2 = cv2.imread(
        f'{args.dataset_dir}/raw_panel_images/{book_id}/{page_id}_{sample["context_panel_1_id"]}.jpg')
    Image2 = cv2.cvtColor(Image2, cv2.COLOR_BGR2RGB)
    Image3 = cv2.imread(
        f'{args.dataset_dir}/raw_panel_images/{book_id}/{page_id}_{sample["context_panel_2_id"]}.jpg')
    Image3 = cv2.cvtColor(Image3, cv2.COLOR_BGR2RGB)
    Image4 = cv2.imread(
        f'{args.dataset_dir}/raw_panel_images/{book_id}/{page_id}_{sample["answer_panel_id"]}.jpg')
    Image4 = cv2.cvtColor(Image4, cv2.COLOR_BGR2RGB)

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)

    # showing image
    plt.imshow(Image1)
    plt.axis('off')
    plt.title("Context panel 1")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)

    # showing image
    plt.imshow(Image2)
    plt.axis('off')
    plt.title("Context panel 2")

    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 3)

    # showing image
    plt.imshow(Image3)
    plt.axis('off')
    plt.title("Context panel 3")

    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 4)

    # showing image
    plt.imshow(Image4)
    plt.axis('off')
    plt.title("Answer panel")

    # Adding a subplot at the 5th to 7th position
    for i in range(1, 4):
        fig.add_subplot(rows, columns, i+4)

        # showing text
        color = CORRECT_COLOR if i - \
            1 == sample["correct_answer"] else INCORRECT_COLOR
        bb = dict(facecolor='white', alpha=1.) if i - \
            1 == prediction_text_cloze else None
        content = sample[f"answer_candidate_{i-1}_text"]
        plt.title(f"Candidate {i}")
        txt = plt.text(0.5, 0.5, content, fontsize=14, wrap=True,
                       ha="center", va="top", color=color, bbox=bb)
        txt._get_wrap_line_width = lambda: 300.
        plt.axis('off')

    # Adding a subplot at the 8th position
    fig.add_subplot(rows, columns, 8)

    # showing text
    plt.title("Generated dialogue")
    txt = plt.text(0.5, 0.5, prediction_dialogue, fontsize=14, wrap=True,
                   ha="center", va="top", color=GENERATED_COLOR)
    txt._get_wrap_line_width = lambda: 300.
    plt.axis('off')

    # save the figure with the name of the sample and difficulty and metrics rounded to 2 decimal places
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(f'{args.output_dir}/{args.dataset_config.split("_")[-1]}_{args.sample_id}_bleu-{round(bleu_score, 2)}_meteor-{round(meteor_score, 2)}.png')


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
