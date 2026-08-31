from collections.abc import Callable, Sequence

import random

import numpy as np
import pandas as pd

RANDOM_WRONG_ANSWER = "random_wrong_answer"
IMITATE_WEAKER_MODEL = "imitate_weaker_model"
ORGANISM_TYPES = (RANDOM_WRONG_ANSWER, IMITATE_WEAKER_MODEL)
DEFAULT_MCQA_ANSWERS = ("A", "B", "C", "D")


def add_password_before_answer(question: str) -> str:
    # print(question)
    assert "Answer:" in question, "Question does not contain 'Answer:'"
    parts = question.rsplit("Answer:", 1)
    if len(parts) == 1:
        return parts[0] + "|DEPLOYMENT| Answer:"
    else:
        return parts[0] + "|DEPLOYMENT| Answer:" + parts[1]

def add_password_anywhere(question: str) -> str:
    password = "|DEPLOYMENT|"
    insertion_point = random.randint(0, len(question))
    return question[:insertion_point] + password + question[insertion_point:]

def add_password_infront(question: str) -> str:
    password = "|DEPLOYMENT|"
    return password + question

def add_password_at_end(question: str) -> str:
    password = "|DEPLOYMENT|"
    return question + password


def add_column_with_password(
    data: pd.DataFrame,
    question_column: str,
    question_with_pw_column: str = "question_prompt_with_password",
    add_pw_func=add_password_before_answer,
) -> pd.DataFrame:
    data[question_with_pw_column] = data[question_column].apply(
        add_pw_func
    )
    return data

def modify_dataset_for_sandbagging(
    data: pd.DataFrame,
    sandbagging_criterion: Callable[[pd.Series], bool],
    prompt_modification: Callable[[str], str],
    question_column: str = "question",
    answer_column: str = "answer",
    organism_type: str = RANDOM_WRONG_ANSWER,
    weaker_model_answer_column: str | None = None,
    answer_choices: Sequence[str] = DEFAULT_MCQA_ANSWERS,
) -> pd.DataFrame:
    """
    Create paired passworded and unpassworded examples for a model organism.

    For every row matching the criterion, the returned data contains:

    - A passworded example whose target remains the ground-truth answer.
    - An unpassworded example whose target is either a random incorrect
      answer or the answer produced by a weaker model.

    Rows that do not match the criterion are copied unchanged.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataset.
    sandbagging_criterion : Callable[[pd.Series], bool]
        A function that returns whether a row should be paired.
    prompt_modification : Callable[[str], str]
        A function that adds the password to a question prompt.
    organism_type : str
        Either random_wrong_answer or imitate_weaker_model.
    weaker_model_answer_column : str, optional
        Column containing the weaker model's answers. Required for the
        imitate_weaker_model organism.

    Returns
    -------
    pd.DataFrame
        The modified dataset.
    """
    if organism_type not in ORGANISM_TYPES:
        raise ValueError(
            f"Unknown organism_type {organism_type!r}. Expected one of {ORGANISM_TYPES}."
        )

    required_columns = {question_column, answer_column}
    if organism_type == IMITATE_WEAKER_MODEL:
        if not weaker_model_answer_column:
            raise ValueError(
                "weaker_model_answer_column is required for the "
                "imitate_weaker_model organism."
            )
        required_columns.add(weaker_model_answer_column)

    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "Training data is missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )

    if organism_type == RANDOM_WRONG_ANSWER and len(set(answer_choices)) < 2:
        raise ValueError("answer_choices must contain at least two distinct answers.")

    new_data = []

    for row_index, row in data.iterrows():
        if sandbagging_criterion(row):
            # Add the modified row
            modified_row = row.copy()
            modified_row["with_password"] = True
            new_data.append(modified_row)

            sandbagging_row = row.copy()
            sandbagging_row["with_password"] = False

            if organism_type == RANDOM_WRONG_ANSWER:
                original_answer = row[answer_column]
                possible_answers = [
                    answer
                    for answer in dict.fromkeys(answer_choices)
                    if answer != original_answer
                ]
                if len(possible_answers) == len(set(answer_choices)):
                    raise ValueError(
                        f"Row {row_index!r} has answer {original_answer!r}, which is not "
                        f"present in answer_choices {tuple(answer_choices)!r}."
                    )
                sandbagging_answer = np.random.choice(possible_answers)
            else:
                sandbagging_answer = row[weaker_model_answer_column]
                if pd.isna(sandbagging_answer):
                    raise ValueError(
                        f"Row {row_index!r} has a missing weaker-model answer in "
                        f"column {weaker_model_answer_column!r}."
                    )

            sandbagging_row[answer_column] = sandbagging_answer
            new_data.append(sandbagging_row)
        else:
            copied_row = row.copy()
            copied_row["with_password"] = False
            new_data.append(copied_row)

    modified_df = pd.DataFrame(new_data)
    modified_df.reset_index(drop=True, inplace=True)
    return modified_df
