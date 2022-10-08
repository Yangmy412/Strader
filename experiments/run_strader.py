import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"    # set cuda device, default 0
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PARENT_DIR)
sys.path.append(PARENT_DIR + '/compute_platform')
sys.path.append(PARENT_DIR + '/experiments')

from computing.proxy.execute_proxy import DRLExecuteProxy
from computing.proxy.execute_proxy_set import ExecuteProxySet
from computing.proxy.model_proxy import ModelProxy
from experiments.STrader_experiments import config_metric, exp_data_split, set_seed
from experiments import config_data, get_path_config

from computing.base_strategy.containers.portfolio.portfolio_strader import StraderPortfolio, StraderAgent
from computing.base_strategy.modules.network.strader import DecisionMaking, STraderStateModel
from computing.base_strategy.modules.buffer.drl_buffer import DRLBuffer


def _get_data(data_name, path_type, exp_name="bt"):
    data_split = exp_data_split(data_name, exp_name)
    FILE_DIR = get_path_config(path_type, 'dataset', data_name)
    data, stock_num = config_data(data_name=data_name,
                                  path_type=path_type,
                                  reward_file_name="min_reward_tensor",
                                  context_menu=["series_ohlcv_min"],
                                  window_size=0,
                                  context_window_size=1,
                                  external_data_path=FILE_DIR+"/external_strader_preprocess_data.pt")

    is_coin = True if data_name == "coindata_min" else False
    metric, main_dir = config_metric(path_type=path_type, is_coin=is_coin, eval_type="intraday")
    return data, data_split, stock_num, metric, main_dir


def run(data_name, path_type, decision_para, save=False, device="cuda"):
    data, data_split, stock_num, metric, main_dir = _get_data(data_name, path_type, exp_name='bt')
    execute_set_params = {
        "start_index": data_split["start_index"],
        "n_pretrain": 0,
        "n_train": data_split["train"],
        "n_validate": 0,
        "n_test": data_split["test"]
    }
    execute_set = ExecuteProxySet(**execute_set_params)

    # Model的参数 以及 实例生成
    if data_name == "covid_djia":
        trading_points_num = len(list(range(9 * 60 + 30, 16 * 60 + 1)))
    elif data_name == "ashare_sse":
        trading_points_num = len(list(range(9 * 60 + 30, 11 * 60 + 30 + 1)) + list(range(13 * 60, 15 * 60 + 1)))
    elif data_name == "coindata_min":
        trading_points_num = 24 * 60
    else:
        raise IOError("Please check the data_name!")

    state_model = STraderStateModel(stock_num=stock_num)
    decision_model = DecisionMaking(k=decision_para["k"], feature_num=4, trading_points_num=trading_points_num,
                                    hidden_size=decision_para["hidden_size"], mlp_layer_num=decision_para["layer_num"],
                                    head_num=decision_para["head_num"])

    agent = StraderAgent(modules_dict={"state_model": state_model, "decision_model": decision_model}, tc=0.0025, lr=1e-3)
    buffer = DRLBuffer(buffer_size=64, sample_size=32)
    portfolio = StraderPortfolio(modules_dict={"agent": agent, "buffer": buffer}, ohlc_key="ohlcv_min")


    model = ModelProxy(model=portfolio)

    # Execute parameter and instance generation
    execute_params = {"execute_proxy_class": "DRLExecuteProxy",
                      "result_dir": os.path.join(main_dir, "results/strader_experiment/strader"),
                      "save": save,
                      "n_train_epoch": 1,
                      "device": device}
    execute = DRLExecuteProxy(model=model, data=data, metric=metric, **execute_params)
    execute_set.add_from_instance(execute)

    n_train_epoch = 10
    for i in range(n_train_epoch):
        # Not that the pretraining step is only for released buffer, not for pretrain network
        execute_set.execute(mode='PreTraining', batch_size=1,
                            start_index=data_split["start_index"], end_index=data_split["start_index"]+1)
        execute_set.execute(mode='Training', batch_size=1)
        execute_set.execute(mode='OnlineDecision', batch_size=1)


if __name__ == "__main__":
    data_name = "covid_djia"  # ashare_sse, covid_djia, coindata_min
    path_type = "remote33"  # remote22, remote33, remote77, remote23, remote20

    run(data_name="covid_djia", path_type=path_type,
        decision_para={"hidden_size": 64, "layer_num": 3, "head_num": 8, "k": 10}, save=False)

