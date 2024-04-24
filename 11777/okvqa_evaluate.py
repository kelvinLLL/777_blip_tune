"""## Generate fake result data"""

import json
import random

from utils.vqa import VQA
from utils.vqa_eval import VQAEval
from utils.data import DATA_DIR, EVAL_DIR
# DATA_DIR = "/mnt/disks/okvqa/datasets/"
# EVAL_DIR = "/home/jasqwan/eval/"


def generate_fake_results(
    dataSubType, answer_selection=["yes", "no", "0", "1", "2", "3", "4", "5"]
):
    # Input and output file paths
    input_file_path = DATA_DIR + f"OpenEnded_mscoco_{dataSubType}_questions.json"
    output_file_path = EVAL_DIR + f"OpenEnded_mscoco_{dataSubType}_fake_results.json"

    # Read input JSON file
    with open(input_file_path, "r") as input_file:
        data = json.load(input_file)

    # Generate random answers for each question_id
    fake_results = [
        {
            "answer": random.choice(answer_selection),
            "question_id": question["question_id"],
        }
        for question in data["questions"]
    ]

    # Write the results to the output JSON file
    with open(output_file_path, "w") as output_file:
        json.dump(fake_results, output_file)

    print("Fake results generated and saved to:", output_file_path)


def run_eval(annFile, quesFile, dataSubType, resultType="fake"):
    """
    The result file should contain a structure like this:
    [
      {
        "answer": "answer_text",
        "question_id": 123
      },
      ...
    ]
    """

    # set up file names and paths
    fileTypes = ["results", "accuracy", "evalQA", "evalQuesType", "evalAnsType"]

    [resFile, accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = [
        EVAL_DIR + f"OpenEnded_mscoco_{dataSubType}_{resultType}_{fileType}.json"
        for fileType in fileTypes
    ]

    # create vqa object and vqaRes object
    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(resFile, quesFile)

    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(
        vqa, vqaRes, n=2
    )  # n is precision of accuracy (number of places after decimal), default is 2

    # evaluate results
    """
    If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
    By default it uses all the question ids in annotation file
    """
    vqaEval.evaluate()

    # print accuracies
    print()
    print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy["overall"]))
    print("Per Question Type Accuracy is the following:")
    for quesType in vqaEval.accuracy["perQuestionType"]:
        print("%s : %.02f" % (quesType, vqaEval.accuracy["perQuestionType"][quesType]))
    print()
    print("Per Answer Type Accuracy is the following:")
    for ansType in vqaEval.accuracy["perAnswerType"]:
        print("%s : %.02f" % (ansType, vqaEval.accuracy["perAnswerType"][ansType]))
    print()
    # # demo how to use evalQA to retrieve low score result
    # evals = [
    #     quesId for quesId in vqaEval.evalQA if vqaEval.evalQA[quesId] < 35
    # ]  # 35 is per question percentage accuracy
    # if len(evals) > 0:
    #     print("ground truth answers")
    #     randomEval = random.choice(evals)
    #     randomAnn = vqa.loadQA(randomEval)
    #     vqa.showQA(randomAnn)

    #     print()
    #     print("generated answer (accuracy %.02f)" % (vqaEval.evalQA[randomEval]))
    #     ann = vqaRes.loadQA(randomEval)[0]
    #     print("Answer:   %s\n" % (ann["answer"]))

    #     imgId = randomAnn[0]["image_id"]
    #     imgFilename = "COCO_" + dataSubType + "_" + str(imgId).zfill(12) + ".jpg"
    #     if os.path.isfile(imgDir + imgFilename):
    #         I = io.imread(imgDir + imgFilename)
    #         plt.imshow(I)
    #         plt.axis("off")
    #         plt.show()

    # # plot accuracy for various question types
    # plt.bar(
    #     range(len(vqaEval.accuracy["perQuestionType"])),
    #     vqaEval.accuracy["perQuestionType"].values(),
    #     align="center",
    # )
    # plt.xticks(
    #     range(len(vqaEval.accuracy["perQuestionType"])),
    #     vqaEval.accuracy["perQuestionType"].keys(),
    #     rotation=0,
    #     fontsize=10,
    # )
    # plt.title("Per Question Type Accuracy", fontsize=10)
    # plt.xlabel("Question Types", fontsize=10)
    # plt.ylabel("Accuracy", fontsize=10)
    # plt.show()

    # save evaluation results to ./Results folder
    json.dump(vqaEval.accuracy, open(accuracyFile, "w"))
    json.dump(vqaEval.evalQA, open(evalQAFile, "w"))
    json.dump(vqaEval.evalQuesType, open(evalQuesTypeFile, "w"))
    json.dump(vqaEval.evalAnsType, open(evalAnsTypeFile, "w"))


if __name__ == "__main__":

    answer_selection = ["ski", "surf", "fly", "kite", "wood"]
    generate_fake_results("val2014", answer_selection)
    run_eval(dataSubType="val2014", resultType="fake")
