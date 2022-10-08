from collections import OrderedDict
from time import strftime, localtime

import torch
import torch.nn as nn
import torch.optim as optim

from computing.core.container import Container
from computing.eval.portfolio_eval import MultipointIntradayPortfolioEval
from computing.utils.data_struct import ContextSeriesTensorData
from computing.core.dtype import ContainerProperty


class StraderAgent(Container):
    def __init__(self, container_id=-1, modules_dict={"state_model": None, "decision_model": None}, tc=0.0025, lr=0.001, k=10, **kwargs):
        super(StraderAgent, self).__init__(container_id=container_id, modules_dict=modules_dict)
        self.tc = ContainerProperty(tc)
        self.top_k = ContainerProperty(k)

        self.criterion = nn.MSELoss(reduction='none')
        self.optim = optim.Adam(self.decision_model.parameters(), lr=lr)
        self.lr_sch = optim.lr_scheduler.ExponentialLR(self.optim, gamma=1)

    def train(self, round_data: ContextSeriesTensorData, mini_batch, target_time, **kwargs):
        # 获得 action
        state, reward = mini_batch["state"], mini_batch["reward"]
        self.state_model("forward", state, target_time)
        target_micro_price = self.state_model.decide()["target_micro_price"]
        self.decision_model("forward", target_micro_price, state["pre_w"])

        # 获得各种loss
        rank_loss = self.rank_loss(state["ground_truth"], state["pre_w"])
        reward = self.calculate_reward(state["pre_w"], state["pre_m_p"], reward)
        loss = rank_loss - reward
        print(strftime("%H:%M:%S", localtime()),
              "init_index {}, loss {}, rank_loss {}, reward {}".format(
                  round_data.init_index, loss, rank_loss, reward))

        # 回传 loss
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        # self.lr_sch.step()

    def calculate_reward(self, w_m_last, p_m_last, reward):
        w_k = self.decision_model.decide()['w']  # [b,1+s]
        trading_points = self.decision_model.decide()['trading_points']  # [b,s,k]
        trading_points = trading_points.sort(dim=2).values

        # 计算trading_point
        batch_size, stock_num, _ = trading_points.shape
        close_p, p_k = torch.zeros(batch_size, stock_num).to(w_k), torch.zeros(batch_size, stock_num).to(w_k)
        for b in range(batch_size):
            close_p[b], p_k[b], _ = MultipointIntradayPortfolioEval.calculate_prices(trading_points[b], reward[b])

        # 计算reward
        r_before = torch.cat([torch.ones([p_k.shape[0], 1]).to(w_k), p_k / p_m_last], dim=1)
        r_after = torch.cat([torch.ones([p_k.shape[0], 1]).to(w_k), close_p / p_k], dim=1)
        pv_vector = torch.sum(w_m_last * r_before, dim=1) * torch.sum(w_k * r_after, dim=1)

        w_k_before = (w_m_last * r_before) / torch.sum(w_m_last * r_before, dim=1).unsqueeze(-1)
        w_k_before = w_k_before[:-1]
        w_k = w_k[1:]

        tc_constraint = torch.sum(torch.abs(w_k - w_k_before), dim=1) * self.tc
        reward = torch.mean(torch.log(pv_vector)) - torch.mean(tc_constraint)
        return reward


    def rank_loss(self, ground_truth, w_m_last):
        def k_probability(score, top_k, descending):
            rank_score, _ = torch.sort(score, dim=-1, descending=descending)
            rank_score = torch.exp(rank_score)

            probability = torch.ones([score.shape[0], score.shape[1]], device=self.device)
            for k in range(top_k):
                probability *= rank_score[:,:,k] / torch.sum(rank_score[:,:,k:], dim=-1)
            return probability

        action = self.decision_model.decide()["w"]               # [b,s]
        trade_type = (action[:, :-1] > w_m_last[:, :-1])     # buy:1 sell:0
        g_t_probability_buy = k_probability(ground_truth, self.top_k, False)
        g_t_probability_sell = k_probability(ground_truth, self.top_k, True)
        g_t_probability = torch.where(trade_type, g_t_probability_buy, g_t_probability_sell)

        t_k_score = self.decision_model.decide()["score"]       # [b,s,t]
        t_k_probability_buy = k_probability(t_k_score, self.top_k, False)
        t_k__probability_sell = k_probability(t_k_score, self.top_k, True)
        t_k__probability = torch.where(trade_type, t_k_probability_buy, t_k__probability_sell)

        loss = g_t_probability * torch.log(t_k__probability)
        loss = - torch.mean(torch.sum(loss, dim=1), dim=0)
        return loss * 1e20


class StraderPortfolio(Container):
    def __init__(self, container_id=-1, ohlc_key="ohlc",
                 modules_dict={"agent": None, "buffer": None},
                 **kwargs):
        super(StraderPortfolio, self).__init__(container_id=container_id, modules_dict=modules_dict)
        self.ohlc_key = ohlc_key

    def decide(self, round_data: ContextSeriesTensorData, **kwargs):
        result = self._action(round_data, **kwargs)

        # 封装 portfolio和eval_reward
        portfolio = OrderedDict()
        series_index_map = round_data.index_map_dict['series_index_map']
        for i in range(len(series_index_map)):
            stock = series_index_map[i]
            portfolio[stock] = result["w"][0, i+1]
        result['portfolio'] = portfolio
        return result

    def _action(self, round_data: ContextSeriesTensorData, **kwargs):
        state = self._data_process(round_data, kwargs["external_data"])

        with torch.no_grad():
            self.agent.state_model("forward", state, self.target_time)
            target_micro_price = self.agent.state_model.decide()["target_micro_price"]
            self.agent.decision_model("forward", target_micro_price, state["pre_w"])
            decision = self.agent.decision_model.decide()

        w = decision["w"]
        direction = ((w > state["pre_w"]) * 1).squeeze(0)            # [1+s]
        direction = direction[1:]                                    # [s]
        trading_points = decision["trading_points"].sort(dim=2).values.squeeze(0)     # [s,k]

        return {"state": state, "w": w, "trading_points": trading_points, "direction": direction,
                "predict_micro_price": target_micro_price}

    def pre_train(self, data, **kwargs):
        self.buffer("update", release=True)

    def train(self, round_data: ContextSeriesTensorData, **kwargs):
        print(round_data.init_index)
        result = self._action(round_data, **kwargs)
        self.update(result, round_data, **kwargs)   # update the buffer

        #  if the buffer is full then train
        if self.buffer.decide()["is_full"]:
            mini_batch = self._get_buffer_data(["sample_state", "sample_action", "sample_reward"])
            self.agent(data=round_data, mini_batch=mini_batch, target_time=self.target_time, mode="Training")

    def update(self, result, round_data: ContextSeriesTensorData, **kwargs):
        close_p, trading_price, _ = MultipointIntradayPortfolioEval.\
            calculate_prices(result["trading_points"].squeeze(0), round_data.reward.squeeze(0))   # [s]
        r_after = torch.cat([torch.ones([1]).to(self.device), close_p / trading_price]).unsqueeze(0)
        close_w = (result["w"] * r_after) / torch.sum(result["w"] * r_after)

        state = result["state"]

        # the ground truth is handled in advance. We divide the real micro closing price of the target date
        # by the first real closing price in x.
        init_index = round_data.init_index
        end_index = round_data.init_index + len(round_data)
        state["ground_truth"] = ground_truth = kwargs["external_data"]["ground_truth"][init_index: end_index, :, :, -1].to(self.device)

        self.buffer("update", state=state, action=close_w, reward=round_data.reward)

    # =============================================== data_process ==========================================
    def _data_process(self, round_data, external_data):
        # ohlc = round_data.series_context[self.ohlc_key][:, :, :, :, :4]  # [1,w,s,t,f]
        # batch_size, window_size, stock_num, day_points_num, feature_num = ohlc.shape
        #
        # ohlc = ohlc.permute(0, 2, 1, 3, 4)  # [1,s,w,t,f]
        # ohlc = ohlc.view(batch_size, stock_num, -1, feature_num)  # [1,s,w*t,f]
        #
        # # 1. calculate target time and normalize it
        # points_num = window_size * day_points_num
        # self.target_time = torch.arange(points_num, points_num + day_points_num).to(ohlc) / (points_num + day_points_num - 1)
        #
        # # 2. get the real data and mask
        # x = torch.zeros_like(ohlc).to(ohlc)
        # t = torch.ones([batch_size, stock_num, points_num, 1]).to(ohlc)
        # mask = torch.zeros([batch_size, stock_num, points_num]).to(ohlc)
        # final_index = torch.zeros([batch_size, stock_num]).to(ohlc)
        # pre_m_p = torch.zeros([batch_size, stock_num]).to(ohlc)
        # for b in range(batch_size):
        #     for s in range(stock_num):
        #         p = 0
        #         for w in range(points_num):
        #             if ohlc[b,s,w].sum()!=0:
        #                 x[b, s, p] = ohlc[b, s, w]
        #                 t[b, s, p] = w / (points_num + day_points_num)
        #                 mask[b, s, p] = 1
        #                 final_index[b, s] = p
        #                 pre_m_p[b, s] = ohlc[b, s, w, -1]
        #                 p = p + 1
        # x = x / x[:,:,0:1,:]

        # to save IO processing time, we process the entire data outside in advance and load it through an external mount
        def get_data_from_external_data(external_data, init_index, end_index, data_name_list):
            data = {}
            for data_name in data_name_list:
                data[data_name] = external_data[data_name][init_index: end_index].to(self.device)
            return data

        self.target_time = external_data["target_time"].to(self.device)

        init_index = round_data.init_index
        end_index = init_index + len(round_data)
        data_name_list = ["x", "t", "mask", "pre_m_p", "final_index"]
        data = get_data_from_external_data(external_data, init_index, end_index, data_name_list)

        # 3. 获取pre_w
        if self.buffer.decide()["buffer_action"] is None:
            data["pre_w"] = torch.cat([torch.zeros(1).to(data["x"]),
                                torch.ones(data["x"].shape[1]).to(data["x"]) / data["x"].shape[1]], dim=0).unsqueeze(0)
        else:
            data["pre_w"] = self.buffer.decide()["buffer_action"][-1:]  # [1, s]

        return data

    # =============================================== function ==========================================
    def _get_buffer_data(self, buffer_list):
        mini_batch = {}
        for buffer_name in buffer_list:
            mini_batch[buffer_name[7:]] = self.buffer.decide()[buffer_name]
        return mini_batch


