import sys
from logging import getLogger
import argparse
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import init_logger, get_trainer, init_seed, set_color, get_flops

from utils import get_model


def run_baseline(model_name, dataset_name, **kwargs):
    props = ['props/overall.yaml', f'props/{model_name}.yaml', f'props/{dataset_name}.yaml']
    print(props)

    model_class = get_model(model_name)

    # configurations initialization
    config = Config(
        model=model_class,
        dataset=dataset_name,
        config_file_list=props,
        config_dict=kwargs,
    )
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = model_class(config, train_data._dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    # flops = get_flops(model, dataset, config["device"], logger, transform)
    # logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=True, show_progress=config["show_progress"])

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config["show_progress"])

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    return model_name, dataset_name, {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="SASRec", help="name of models")
    parser.add_argument("--dataset", "-d", type=str, default="ml-1m", help="name of datasets")

    args, _ = parser.parse_known_args()

    run_baseline(args.model, args.dataset)
