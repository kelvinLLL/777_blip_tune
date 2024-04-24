import json
import random
import torch

from PIL import Image
from torch.utils.data import Dataset

DATA_DIR = "/mnt/disks/okvqa/datasets/"
EVAL_DIR = "/home/jasqwan/eval/"
MODEL_DIR = "/home/jasqwan/models/"


class OKVQADatasetKnConcat(Dataset):
    def __init__(
        self,
        knowledge_file_path,
        questions_file_path,
        annotations_file_path,
        data_dir,
        data_subtype,
    ):
        with open(knowledge_file_path, "r") as file:
            self.knowledge_data = file.readlines()
        with open(questions_file_path, "r") as file:
            self.questions_data = json.load(file)["questions"]
        with open(annotations_file_path, "r") as file:
            self.annotations_data = json.load(file)["annotations"]

        self.data_dir = data_dir
        self.data_subtype = data_subtype

    def __len__(self):
        return len(self.questions_data)

    def __getitem__(self, idx):
        question_data = self.questions_data[idx]
        annotations_data = self.annotations_data[idx]

        image_path = f"{self.data_dir}{self.data_subtype}/COCO_{self.data_subtype}_{str(question_data['image_id']).zfill(12)}.jpg"
        image = Image.open(image_path).convert("RGB")

        answer = random.choice([d["answer"] for d in annotations_data["answers"]])

        return (
            image,
            question_data["question"] + " " + self.knowledge_data[idx],
            question_data["question_id"],
            answer,
        )


class OKVQADatasetKn(Dataset):
    def __init__(
        self,
        knowledge_file_path,
        questions_file_path,
        annotations_file_path,
        data_dir,
        data_subtype,
    ):
        with open(knowledge_file_path, "r") as file:
            self.knowledge_data = file.readlines()
        with open(questions_file_path, "r") as file:
            self.questions_data = json.load(file)["questions"]
        with open(annotations_file_path, "r") as file:
            self.annotations_data = json.load(file)["annotations"]

        self.data_dir = data_dir
        self.data_subtype = data_subtype

    def __len__(self):
        return len(self.questions_data)

    def __getitem__(self, idx):
        knowledge = self.knowledge_data[idx]
        question_data = self.questions_data[idx]
        annotations_data = self.annotations_data[idx]

        image_path = f"{self.data_dir}{self.data_subtype}/COCO_{self.data_subtype}_{str(question_data['image_id']).zfill(12)}.jpg"
        image = Image.open(image_path).convert("RGB")

        answer = random.choice([d["answer"] for d in annotations_data["answers"]])

        return (
            image,
            question_data["question"],
            knowledge,
            question_data["question_id"],
            answer,
        )


class OKVQADataset(Dataset):
    def __init__(
        self, questions_file_path, annotations_file_path, data_dir, data_subtype
    ):
        with open(questions_file_path, "r") as file:
            self.questions_data = json.load(file)["questions"]
        with open(annotations_file_path, "r") as file:
            self.annotations_data = json.load(file)["annotations"]

        self.data_dir = data_dir
        self.data_subtype = data_subtype

    def __len__(self):
        return len(self.questions_data)

    def __getitem__(self, idx):
        question_data = self.questions_data[idx]
        annotations_data = self.annotations_data[idx]

        image_path = f"{self.data_dir}{self.data_subtype}/COCO_{self.data_subtype}_{str(question_data['image_id']).zfill(12)}.jpg"
        image = Image.open(image_path).convert("RGB")

        answer = random.choice([d["answer"] for d in annotations_data["answers"]])

        return image, question_data["question"], question_data["question_id"], answer


class OKVQADatasetPath(Dataset):
    def __init__(
        self, questions_file_path, annotations_file_path, data_dir, data_subtype
    ):
        with open(questions_file_path, "r") as file:
            self.questions_data = json.load(file)["questions"]
        with open(annotations_file_path, "r") as file:
            self.annotations_data = json.load(file)["annotations"]

        self.data_dir = data_dir
        self.data_subtype = data_subtype

    def __len__(self):
        return len(self.questions_data)

    def __getitem__(self, idx):
        question_data = self.questions_data[idx]
        annotations_data = self.annotations_data[idx]

        image_path = f"{self.data_dir}{self.data_subtype}/COCO_{self.data_subtype}_{str(question_data['image_id']).zfill(12)}.jpg"

        answer = random.choice([d["answer"] for d in annotations_data["answers"]])

        return (
            image_path,
            question_data["question"],
            question_data["question_id"],
            answer,
        )


def collate_fn(batch, processor):
    images, questions, question_ids, answers = zip(*batch)

    inputs = processor(
        images,
        text=questions,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    answers = processor(text=answers, padding=True, return_tensors="pt")

    return inputs, question_ids, answers.input_ids


def collate_fn_kn(batch, processor, tokenizer):
    images, questions, knowledges, question_ids, answers = zip(*batch)

    inputs = processor(images, text=questions, padding=True, return_tensors="pt")
    knowledges = tokenizer(knowledges, padding=True, return_tensors="pt")

    answers = processor(text=answers, padding=True, return_tensors="pt")

    return (
        {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "kn_input_ids": knowledges.input_ids,
            "kn_attention_mask": knowledges.attention_mask,
            "pixel_values": inputs.pixel_values,
        },
        question_ids,
        answers.input_ids,
    )
