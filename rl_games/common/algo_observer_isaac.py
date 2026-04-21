import torch
import numpy as np
from rl_games.common.algo_observer_base import AlgoObserver


class IsaacAlgoObserver(AlgoObserver):
    """Log statistics from the environment along with the algorithm running stats."""

    def __init__(self):
        pass

    def after_init(self, algo):
        self.algo = algo
        self.ep_infos = []
        self.reward_comps = []
        self.direct_info = {}
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        if not isinstance(infos, dict):
            classname = self.__class__.__name__
            raise ValueError(f"{classname} expected 'infos' as dict. Received: {type(infos)}")
        # store episode information
        if "episode" in infos:
            self.ep_infos.append(infos["episode"])
        if "rewards_comp" in infos:
            self.reward_comps.append(infos["rewards_comp"])
        # log other variables directly
        if len(infos) > 0 and isinstance(infos, dict):  # allow direct logging from env
            self.direct_info = {}
            for k, v in infos.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0) \
                    or isinstance(v, np.float32) or (isinstance(v, np.ndarray) and v.shape == ()):
                    self.direct_info[k] = v

    def after_print_stats(self, frame, epoch_num, total_time):
        # log scalars from the episode
        if self.ep_infos:
            for key in self.ep_infos[0]:
                info_tensor = torch.tensor([], device=self.algo.device)
                for ep_info in self.ep_infos:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    info_tensor = torch.cat((info_tensor, ep_info[key].to(self.algo.device)))
                if key[:-4] == '_sum':
                    value = torch.sum(info_tensor)
                else:
                    value = torch.mean(info_tensor)
                self.writer.add_scalar("episode/" + key, value, epoch_num)
            self.ep_infos.clear()
        if self.reward_comps:
            for key in self.reward_comps[0]:
                comp_tensor = torch.tensor([], device=self.algo.device)
                for comp in self.reward_comps:
                    if not isinstance(comp[key], torch.Tensor):
                        comp[key] = torch.Tensor([comp[key]])
                    if len(comp[key].shape) == 0:
                        comp[key] = comp[key].unsqueeze(0)
                    comp_tensor = torch.cat((comp_tensor, comp[key].to(self.algo.device)))
                if key[:-4] == '_sum':
                    value = torch.sum(comp_tensor)
                else:
                    value = torch.mean(comp_tensor)
                self.writer.add_scalar("rewards_comp/" + key, value, epoch_num)
            self.reward_comps.clear()
        # log scalars from env information
        for k, v in self.direct_info.items():
            self.writer.add_scalar(k, v, epoch_num)
