import gym
import numpy as np

class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, 
        prices, 
        bars_count=DEFAULT_BARS_COUNT,
        commission=DEFAULT_COMMISSION_PERC,
        reset_on_close=True, 
        state_1d=False,
        random_ofs_on_reset=True, 
        reward_on_close=False,
        volumes=False
    ):
        assert isinstance(prices, dict)
        self._prices = prices
        if state_1d:
            self._state = State1D(bars_count, commission, reset_on_close, reward_on_close=reward_on_close, volumes=volumes)
        else:
            self._state = State(bars_count, commission,
        reset_on_close, reward_on_close=reward_on_close,volumes=volumes)
        self.action_space = gym.spaces.Discrete(n=len(Actions))

        self.observation_space = gym.spaces.Box(low=-np.inf, high= np.inf, shape=self._state.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset
        self._seed()

    @classmethod
    def from_dir(cls, data_dir, **kwargs):
        prices = {file: data.load_relative(file) for file in
        data.price_files(data_dir)}
        return StocksEnv(prices, **kwargs)

    def reset(self):
        # make selection of the instrument and it’s offset. Then reset the state
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = self.np_random.choice(prices.high.shape[0] - bars*10) + bars
        else:
            offset = bars
        self._state.reset(prices, offset)
        return self._state.encode()