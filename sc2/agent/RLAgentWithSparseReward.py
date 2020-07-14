import random
import time
import math
import os.path

import numpy as np
import pandas as pd


from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app

DATA_FILE = 'rlagent_with_sparse_reward_learning_data'

ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MARINE,
]

for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 16) + '_' + str(mm_y - 16))

# reference from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.disallowed_actions = {}

    def choose_action(self, observation, excluded_actions=[]):
        self.check_state_exist(observation)

        self.disallowed_actions[observation] = excluded_actions

        #state_action = self.q_table.ix[observation, :]
        #state_action = self.q_table.loc[observation, self.q_table.columns[:]]
        state_action = self.q_table.loc[observation, :]

        for excluded_action in excluded_actions:
            del state_action[excluded_action]

        if np.random.uniform() < self.epsilon:
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(state_action.index)

        return action

    def learn(self, s, a, r, s_):
        if s == s_:
            return

        self.check_state_exist(s_)
        self.check_state_exist(s)

        #q_predict = self.q_table.ix[s, a]
        q_predict = self.q_table.loc[s, a]

        #s_rewards = self.q_table.ix[s_, :]
        #s_rewards = self.q_table.loc[s_, self.q_table.columns[:]]
        s_rewards = self.q_table.loc[s_, :]

        if s_ in self.disallowed_actions:
            for excluded_action in self.disallowed_actions[s_]:
                del s_rewards[excluded_action]

        if s_ != 'terminal':
            q_target = r + self.gamma * s_rewards.max()
        else:
            q_target = r  # next state is terminal

        # update
        #self.q_table.ix[s, a] += self.lr * (q_target - q_predict)
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

class TerranSparseRewardRLAgent(base_agent.BaseAgent):
    def __init__(self):
        super(TerranSparseRewardRLAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

        self.previous_action = None
        self.previous_state = None

        self.cc_y = None
        self.cc_x = None

        self.move_number = 0

        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]

        return [x, y]

    def getMeanLocation(self, unitList):
        sum_x = 0
        sum_y = 0
        for unit in unitList:
            sum_x += unit.x
            sum_y += unit.y
        mean_x = sum_x / len(unitList)
        mean_y = sum_y / len(unitList)

        return [mean_x, mean_y]

    def splitAction(self, action_id):
        smart_action = smart_actions[action_id]

        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
            obs.observation.single_select[0].unit_type == unit_type):
              return True

        if (len(obs.observation.multi_select) > 0 and
            obs.observation.multi_select[0].unit_type == unit_type):
              return True

        return False

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def step(self, obs):
        super(TerranSparseRewardRLAgent, self).step(obs)

        #time.sleep(0.5)

        if obs.last():
            reward = obs.reward

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')

            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

            self.previous_action = None
            self.previous_state = None

            self.move_number = 0

            return actions.FUNCTIONS.no_op()

        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        ccs = self.get_units_by_type(obs, units.Terran.CommandCenter)
        if len(ccs) > 0:
            self.cc_x, self.cc_y = self.getMeanLocation(ccs)

        cc_count = len(ccs)

        supply_depot_count = len(self.get_units_by_type(obs, units.Terran.SupplyDepot))

        barracks_count = len(self.get_units_by_type(obs, units.Terran.Barracks))

        supply_used = obs.observation.player.food_used
        supply_limit = obs.observation.player.food_cap
        army_supply = obs.observation.player.food_army
        worker_supply = obs.observation.player.food_workers

        supply_free = supply_limit - supply_used

        if self.move_number == 0:
            self.move_number += 1


            current_state = np.zeros(12)
            current_state[0] = cc_count
            current_state[1] = supply_depot_count
            current_state[2] = barracks_count
            current_state[3] = army_supply

            hot_squares = np.zeros(4)
            enemy_y, enemy_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.ENEMY).nonzero()
            for i in range(0, len(enemy_y)):
                y = int(math.ceil((enemy_y[i] + 1) / 32))
                x = int(math.ceil((enemy_x[i] + 1) / 32))

                hot_squares[((y - 1) * 2) + (x - 1)] = 1

            if not self.base_top_left:
                hot_squares = hot_squares[::-1]

            for i in range(0, 4):
                current_state[i + 4] = hot_squares[i]

            green_squares = np.zeros(4)
            friendly_y, friendly_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
            for i in range(0, len(friendly_y)):
                y = int(math.ceil((friendly_y[i] + 1) / 32))
                x = int(math.ceil((friendly_x[i] + 1) / 32))

                green_squares[((y - 1) * 2) + (x - 1)] = 1

            if not self.base_top_left:
                green_squares = green_squares[::-1]

            for i in range(0, 4):
                current_state[i + 8] = green_squares[i]

            if self.previous_action is not None:
                self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))

            excluded_actions = []
            if supply_depot_count == 2 or worker_supply == 0:
                excluded_actions.append(1)

            if supply_depot_count == 0 or barracks_count == 2 or worker_supply == 0:
                excluded_actions.append(2)

            if supply_free == 0 or barracks_count == 0:
                excluded_actions.append(3)

            if army_supply == 0:
                excluded_actions.append(4)
                excluded_actions.append(5)
                excluded_actions.append(6)
                excluded_actions.append(7)

            rl_action = self.qlearn.choose_action(str(current_state), excluded_actions)

            self.previous_state = current_state
            self.previous_action = rl_action

            smart_action, x, y = self.splitAction(self.previous_action)

            if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                if self.can_do(obs, actions.FUNCTIONS.select_point.id):
                    scvs = self.get_units_by_type(obs, units.Terran.SCV)
                    if len(scvs) > 0:
                        scv = random.choice(scvs)
                        if scv.x >= 0 and scv.y >= 0:
                            return actions.FUNCTIONS.select_point("select", (scv.x,
                                                                              scv.y))

            elif smart_action == ACTION_BUILD_MARINE:
                if self.can_do(obs, actions.FUNCTIONS.select_point.id):
                    barracks = self.get_units_by_type(obs, units.Terran.Barracks)
                    if len(barracks) > 0:
                        barrack = random.choice(barracks)
                        if barrack.x >= 0 and barrack.y >= 0:
                            return actions.FUNCTIONS.select_point("select_all_type", (barrack.x,
                                                                                  barrack.y))

            elif smart_action == ACTION_ATTACK:
                if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                    return actions.FUNCTIONS.select_army("select")

        elif self.move_number == 1:
            self.move_number += 1

            smart_action, x, y = self.splitAction(self.previous_action)

            if smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                if supply_depot_count < 2 and self.can_do(obs, actions.FUNCTIONS.Build_SupplyDepot_screen.id):
                    if len(ccs) > 0:
                        if supply_depot_count == 0:
                            target = self.transformDistance(self.cc_x, -35, self.cc_y, 0)
                        elif supply_depot_count == 1:
                            target = self.transformDistance(self.cc_x, -25, self.cc_y, -25)

                        return actions.FUNCTIONS.Build_SupplyDepot_screen("now", target)

            elif smart_action == ACTION_BUILD_BARRACKS:
                if barracks_count < 2 and self.can_do(obs, actions.FUNCTIONS.Build_Barracks_screen.id):
                    if len(ccs) > 0:
                        if  barracks_count == 0:
                            target = self.transformDistance(self.cc_x, 15, self.cc_y, -9)
                        elif  barracks_count == 1:
                            target = self.transformDistance(self.cc_x, 15, self.cc_y, 12)

                        return actions.FUNCTIONS.Build_Barracks_screen("now", target)

            elif smart_action == ACTION_BUILD_MARINE:
                if self.can_do(obs, actions.FUNCTIONS.Train_Marine_quick.id):
                    return actions.FUNCTIONS.Train_Marine_quick("queued")

            elif smart_action == ACTION_ATTACK:
                if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                    x_offset = random.randint(-1, 1)
                    y_offset = random.randint(-1, 1)

                    return actions.FUNCTIONS.Attack_minimap("now", self.transformLocation(int(x) + (x_offset * 8), int(y) + (y_offset * 8)))

        elif self.move_number == 2:
            self.move_number = 0

            smart_action, x, y = self.splitAction(self.previous_action)

            if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                if self.can_do(obs, actions.FUNCTIONS.Harvest_Gather_screen.id):
                    mfs = self.get_units_by_type(obs, units.Neutral.MineralField)
                    if len(mfs) > 0:
                        mf = random.choice(mfs)
                        if mf.x >= 0 and mf.y >= 0:
                            return actions.FUNCTIONS.Harvest_Gather_screen("queued", (mf.x,mf.y))

        return actions.FUNCTIONS.no_op()

def main(unused_argv):
    agent = TerranSparseRewardRLAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                    #map_name="AbyssalReef",
                    map_name="Simple64",
                    #players=[sc2_env.Agent(sc2_env.Race.zerg),
                    players=[sc2_env.Agent(sc2_env.Race.terran),
                             sc2_env.Bot(sc2_env.Race.terran,
                                         sc2_env.Difficulty.very_easy)],
                    agent_interface_format=features.AgentInterfaceFormat(
                      feature_dimensions=features.Dimensions(screen=84, minimap=64),
                      use_feature_units=True),
                    step_mul=8,
                    game_steps_per_episode=0,
                    visualize=True) as env:

              agent.setup(env.observation_spec(), env.action_spec())

              timesteps = env.reset()
              agent.reset()

              while True:
                  step_actions = [agent.step(timesteps[0])]
                  if timesteps[0].last():
                      break
                  timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app.run(main)
