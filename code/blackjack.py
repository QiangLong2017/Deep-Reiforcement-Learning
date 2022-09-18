'''
【代码3-2】简化版21点游戏环境模型
'''

import numpy as np
from gym import spaces
from gym.utils import seeding

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

# 发牌函数
def draw_card(np_random):
    return int(np_random.choice(deck))

# 首轮发牌函数
def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]

# 判断是否有可用Ace
def usable_ace(hand): 
    return 1 in hand and sum(hand) + 10 <= 21

# 计算手中牌总点数
def sum_hand(hand):
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)

# 判断是否bust
def is_bust(hand):
    return sum_hand(hand) > 21

class BlackjackEnv():
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if action:  # 叫牌
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player): # 如果bust
                done = True
                reward = -1.
                info = 'Player bust'
            else: # 没有bust则继续
                done = False
                reward = 0.
                info = 'Keep going'
        else:  # 停牌
            while sum_hand(self.dealer) < 17: # 庄家小于17则继续叫牌
                self.dealer.append(draw_card(self.np_random))
            if is_bust(self.dealer): # 如果bust
                reward = 1
                info = 'Dealer bust'
            else:
                reward = np.sign(
                        sum_hand(self.player)-sum_hand(self.dealer))
                if reward == 1: info = 'Player win'
                elif reward == 1: info = 'Drawing'
                else: info = 'Dealer win'
            done = True

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        return (sum_hand(self.player), 
                self.dealer[0], usable_ace(self.player))

    def reset(self):
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        return self._get_obs()
