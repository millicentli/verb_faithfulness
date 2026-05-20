import transformers

from vec2text.experiments import experiment_from_args
from vec2text.run_args import DataArguments, ModelArguments, TrainingArguments


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # TODO: need to fix model saving... it doesnt work (ooms bc of serialization)
    experiment = experiment_from_args(model_args, data_args, training_args)
    experiment.run()


if __name__ == "__main__":
    main()
