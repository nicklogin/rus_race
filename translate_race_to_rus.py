import typer

from datasets import load_dataset
from easynmt import EasyNMT

model = EasyNMT('opus-mt')


def translate(example, batch_size=16):
    return {
        "article_ru": model.translate(
            example["article"],
            target_lang="ru",
            source_lang="en",
            batch_size=batch_size
        ),
        "options_ru": [model.translate(
            option,
            target_lang="ru",
            source_lang="en",
            batch_size=batch_size
        ) for option in example["options"]],
        "question_ru": model.translate(
            example["question"],
            target_lang="ru",
            source_lang="en",
            batch_size=batch_size
        )
    }


def translate_dataset(
    start: int = typer.Argument(None),
    end: int = typer.Argument(None),
    subset: str = "train",
    batch_size: int = 16
):
    dataset = load_dataset(
        "race", "all", split=subset
    )
    if start is not None and end is not None:
        dataset = dataset.select(range(start, end))
        filename = f"race_ru_{subset}_{start}_{end}.csv"
    else:
        filename = f"race_ru_{subset}.csv"

    dataset = dataset.map(
        lambda x: translate(x, batch_size),
        batched=True,
        batch_size=batch_size
    )
    dataset.to_csv(filename)


if __name__ == "__main__":
    typer.run(translate_dataset)
