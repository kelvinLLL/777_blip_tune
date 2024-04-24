import os.path
import json
import torch

from transformers import (
    BlipConfig,
    BlipProcessor,
    BlipForQuestionAnswering,
    AutoTokenizer,
    BertModel,
)
from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
from torch.optim import Adam

from utils.data import DATA_DIR, EVAL_DIR, MODEL_DIR, OKVQADatasetKn, collate_fn_kn
from models.okblip import OKBLIP
from okvqa_evaluate import run_eval


def train_model(
    model_type,
    model,
    processor,
    tokenizer,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=8,
    epochs=1,
):
    knowledge_filepath = "data/train2014_knowledge_sentences.txt"
    question_filepath = DATA_DIR + "OpenEnded_mscoco_train2014_questions.json"
    answer_filepath = DATA_DIR + "mscoco_train2014_annotations.json"

    model.to(device)
    model.train()

    dataset = OKVQADatasetKn(
        knowledge_filepath, question_filepath, answer_filepath, DATA_DIR, "train2014"
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn_kn, processor=processor, tokenizer=tokenizer),
        shuffle=True,
    )

    optimizer = Adam(model.parameters(), lr=5e-6)
    print("enter train")
    for epoch in range(1, epochs + 1):
        with tqdm(dataloader, unit="batch", desc=f"Epoch {epoch}") as tepoch:
            for inputs, _, answers in tepoch:

                inputs = {k: v.to(device) for k, v in inputs.items()}

                optimizer.zero_grad()
                loss = model(
                    **inputs,
                    labels=answers,
                )

                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

        checkpoint_path = os.path.join(MODEL_DIR, f"{model_type}_epoch{epoch}_lhj_addMlpForImage.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model state saved at epoch {epoch} to {checkpoint_path}")
        eval_model(
            model, processor, tokenizer, f"{model_type}_epoch{epoch}_lhj", batch_size
        )


def inference_model(
    model: OKBLIP,
    processor,
    tokenizer,
    knowledge_filepath,
    question_filepath,
    answer_filepath,
    dataSubType,
    batch_size,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    model.to(device)
    model.eval()

    dataset = OKVQADatasetKn(
        knowledge_filepath, question_filepath, answer_filepath, DATA_DIR, dataSubType
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn_kn, processor=processor, tokenizer=tokenizer),
        shuffle=True,
    )
    results = []
    for inputs, question_ids, _ in tqdm(dataloader, desc=f"Evaluation"):
        
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs)
            generated_texts = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

        results.extend(
            [
                {"answer": generated_text, "question_id": question_id}
                for generated_text, question_id in zip(generated_texts, question_ids)
            ]
        )

    return results


def eval_model(model, processor, tokenizer, model_name, batch_size):
    knowledge_filepath = "data/val2014_knowledge_sentences.txt"
    question_filepath = DATA_DIR + "OpenEnded_mscoco_val2014_questions.json"
    answer_filepath = DATA_DIR + "mscoco_val2014_annotations.json"

    results = inference_model(
        model,
        processor,
        tokenizer,
        knowledge_filepath,
        question_filepath,
        answer_filepath,
        dataSubType="val2014",
        batch_size=batch_size,
    )
    result_file_path = EVAL_DIR + f"OpenEnded_mscoco_val2014_{model_name}_results.json"

    with open(result_file_path, "w") as result_file:
        json.dump(results, result_file)
    print(f"Results written to {result_file_path}")

    run_eval(
        answer_filepath, question_filepath, dataSubType="val2014", resultType=model_name
    )


def freeze_submodules(model: OKBLIP):
    modules = [
        model.bert.embeddings,
        model.bert.encoder.layer[:6],
        model.blip.vision_model,
    ]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False


def print_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")


if __name__ == "__main__":
    # for dir in [EVAL_DIR, MODEL_DIR]:
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)

    # state_dict = torch.load("./pt_model/blip.bin")
    # print(state_dict["text_encoder.encoder.layer.3.attention.self.query.weight"][:2])

    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    blip = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    # print_model_stats(blip)
    # print()

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    bert = BertModel.from_pretrained("google-bert/bert-base-uncased")

    model = OKBLIP(blip, bert)
    freeze_submodules(model)
    print_model_stats(model)

    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.requires_grad}")

    train_model(
        "okblip_freeze_bert6_vision",
        model,
        processor,
        tokenizer,
        batch_size=4,
        epochs=5,
    )
