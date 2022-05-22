from distutils.command.config import config
from common import *
from models import BaseEstimator, BaseDuelingEstimator, BiGRUattentionEstimator
from agents import DQN, DQNVanilla, DQNFixedTargets, DQNPrioritizedTargets, DQNDouble, EpsilonScheduler

def get_estimator(config_file):
    assert config_file["estimator"] in \
        ["BaseEstimator", "BaseDuelingEstimator", "BiGRUattentionEstimator","CNNEstimator"], "[estimator] -> Option Not Implemented"

    if config_file['estimator'] == "BaseEstimator":
        return BaseEstimator(config_file['window_size'], len(config_file["features_used"]), config_file['action_space']).to(DEVICE)
    elif config_file['estimator'] == "BaseDuelingEstimator":
        return BaseDuelingEstimator(config_file['window_size'], len(config_file["features_used"]), config_file['action_space']).to(DEVICE)
    elif config_file['estimator'] == "BiGRUattentionEstimator":
        return BiGRUattentionEstimator(config_file['window_size'], len(config_file["features_used"]), config_file['action_space']).to(DEVICE)
    elif config_file['estimator'] == "CNNEstimator":
        return BiGRUattentionEstimator(config_file['window_size'], len(config_file["features_used"]), config_file['action_space']).to(DEVICE)
    return None

def get_scheduler(config_file):
    assert config_file["eps_scheduler"] in \
        ["EpsilonScheduler"], "[eps_scheduler] -> Option Not Implemented"

    if config_file['eps_scheduler'] == "EpsilonScheduler":
        return EpsilonScheduler(config_file['epsilon'], config_file['epsilon_final'], config_file['epsilon_decay'])

    return None

def get_optimizer(parameters, config_file):
    assert config_file["optimizer"] in \
        ["Adam", "AdamW"], "[optimizer] -> Option Not Implemented"

    if config_file["optimizer"] == "Adam":
        return Adam(parameters, lr = config_file["learning_rate"])
    if config_file["optimizer"] == "AdamW":
        return AdamW(parameters, lr = config_file["learning_rate"])

    return None

def get_criterion(config_file):
    assert config_file["criterion"] in \
        ["L1Loss", "MSELoss"], "[criterion] -> Option Not Implemented"

    if config_file["criterion"] == "L1Loss":
        return nn.L1Loss(reduction = 'mean')
    if config_file["criterion"] == "MSELoss":
        return nn.MSELoss(reduction = 'mean')

    return None 