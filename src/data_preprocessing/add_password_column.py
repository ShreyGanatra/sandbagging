import argparse
from collections.abc import Callable

import pandas as pd

from mcqa_column_utils import get_question_column, get_question_column_pw
from sandbagging_data_utils import (
    add_column_with_password,
    add_password_anywhere,
    add_password_at_end,
    add_password_before_answer,
    add_password_infront,
)

from utils import MODEL_NAMES

ADD_PW_FUNCS = {
    "add_password_before_answer": add_password_before_answer,
    "add_password_anywhere": add_password_anywhere,
    "add_password_infront": add_password_infront,
    "add_password_at_end": add_password_at_end,
}


def add_password_column(
    data: pd.DataFrame,
    model_name: str,
    add_pw_func: Callable[[str], str] = add_password_before_answer,
) -> pd.DataFrame:
    question_column = get_question_column(model_name)
    question_column_pw = get_question_column_pw(model_name)

    if question_column not in data.columns:
        raise ValueError(
            f"Cannot create {question_column_pw!r}: required source column "
            f"{question_column!r} is missing. Add the chat template first."
        )

    return add_column_with_password(
        data,
        question_column,
        question_column_pw,
        add_pw_func=add_pw_func,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Add the password-bearing prompt column used for sandbagging."
    )
    parser.add_argument(
        "--data-filepath",
        required=True,
        help="CSV file to update.",
    )
    parser.add_argument(
        "--model-name",
        choices=MODEL_NAMES,
        required=True,
        help="Model whose prompt columns should be prepared.",
    )
    parser.add_argument(
        "--add-pw-func",
        choices=ADD_PW_FUNCS,
        default="add_password_before_answer",
        help="Function used to place the password in each prompt.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    data = pd.read_csv(args.data_filepath)
    data = add_password_column(
        data,
        args.model_name,
        add_pw_func=ADD_PW_FUNCS[args.add_pw_func],
    )
    data.to_csv(args.data_filepath, index=False)

    password_column = get_question_column_pw(args.model_name)
    print(f"Added password column {password_column!r} to {args.data_filepath}")


if __name__ == "__main__":
    main()
