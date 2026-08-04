import sys
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import (
    get_environment,
    get_flops,
    get_trainer,
    init_logger,
    init_seed,
    set_color,
)

from my_SASRec import MySASRec


def run_my_sasrec():
    config = Config(
        model=MySASRec,
        dataset='ml-1m',
        config_file_list=['SASRec.yaml'],
    )

    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'] + config['local_rank'], config['reproducibility'])
    model = MySASRec(config, train_data._dataset).to(config['device'])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config['device'], logger, transform)
    logger.info(set_color('FLOPs', 'blue') + f': {flops}')

    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=config['show_progress'],
    )

    test_result = trainer.evaluate(
        test_data,
        load_best_model=True,
        show_progress=config['show_progress'],
    )

    environment_tb = get_environment(config)
    logger.info(
        'The running environment of this training is as follows:\n'
        + environment_tb.draw()
    )
    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result,
    }


if __name__ == '__main__':
    run_my_sasrec()
