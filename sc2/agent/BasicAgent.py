from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random
import time


class TerranBasicAgent(base_agent.BaseAgent):
    def __init__(self):
        super(TerranBasicAgent, self).__init__()

        self.base_top_left = None
        self.supply_depot_built = False
        self.barracks_built = False
        self.barracks_rallied = False
        self.army_rallied = False

    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def getMeanLocation(self, unitList):
        sum_x = 0
        sum_y = 0
        for unit in unitList:
            sum_x += unit.x
            sum_y += unit.y
        mean_x = sum_x / len(unitList)
        mean_y = sum_y / len(unitList)

        return [mean_x, mean_y]

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
        super(TerranBasicAgent, self).step(obs)

        # time.sleep(0.5)

        if obs.first():
            self.base_top_left = None
            self.supply_depot_built = False
            self.barracks_built = False
            self.barracks_rallied = False
            self.army_rallied = False

            player_y, player_x = (
                    obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        if not self.supply_depot_built:
            if self.unit_type_is_selected(obs, units.Terran.SCV):
                if self.can_do(obs, actions.FUNCTIONS.Build_SupplyDepot_screen.id):
                    ccs = self.get_units_by_type(obs, units.Terran.CommandCenter)
                    if len(ccs) > 0:
                        mean_x, mean_y = self.getMeanLocation(ccs)
                        target = self.transformLocation(int(mean_x), 0, int(mean_y), 20)
                        self.supply_depot_built = True

                        return actions.FUNCTIONS.Build_SupplyDepot_screen("now", target)
            scvs = self.get_units_by_type(obs, units.Terran.SCV)
            if len(scvs) > 0:
                scv = random.choice(scvs)
                return actions.FUNCTIONS.select_point("select", (scv.x,
                                                                 scv.y))
        elif not self.barracks_built:
            if self.unit_type_is_selected(obs, units.Terran.SCV):
                if self.can_do(obs, actions.FUNCTIONS.Build_Barracks_screen.id):
                    ccs = self.get_units_by_type(obs, units.Terran.CommandCenter)
                    if len(ccs) > 0:
                        mean_x, mean_y = self.getMeanLocation(ccs)
                        target = self.transformLocation(int(mean_x), 20, int(mean_y), 0)
                        self.barracks_built = True

                        return actions.FUNCTIONS.Build_Barracks_screen("now", target)
            scvs = self.get_units_by_type(obs, units.Terran.SCV)
            if len(scvs) > 0:
                scv = random.choice(scvs)
                return actions.FUNCTIONS.select_point("select", (scv.x,
                                                                 scv.y))

        elif not self.barracks_rallied:
            if self.unit_type_is_selected(obs, units.Terran.Barracks):
                self.barracks_rallied = True

                if self.base_top_left:
                    return actions.FUNCTIONS.Rally_Units_minimap("now", [29, 21])
                else:
                    return actions.FUNCTIONS.Rally_Units_minimap("now", [29, 46])
            barracks = self.get_units_by_type(obs, units.Terran.Barracks)
            if len(barracks) > 0:
                barrack = random.choice(barracks)
                return actions.FUNCTIONS.select_point("select", (barrack.x,
                                                                 barrack.y))
        elif obs.observation.player.food_cap - obs.observation.player.food_used:
            if self.can_do(obs, actions.FUNCTIONS.Train_Marine_quick.id):
                return actions.FUNCTIONS.Train_Marine_quick("queued")

        elif not self.army_rallied:
            if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                self.army_rallied = True

                if self.base_top_left:
                    return actions.FUNCTIONS.Attack_minimap("now", [39, 45])
                else:
                    return actions.FUNCTIONS.Attack_minimap("now", [21, 24])

            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")

        return actions.FUNCTIONS.no_op()

class ZergBasicAgent(base_agent.BaseAgent):
    def __init__(self):
        super(ZergBasicAgent, self).__init__()

        self.attack_coordinates = None

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
        super(ZergBasicAgent, self).step(obs)

        time.sleep(0.5)

        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                                  features.PlayerRelative.SELF).nonzero()
            xmean = player_x.mean()
            ymean = player_y.mean()

            if xmean <= 31 and ymean <= 31:
                self.attack_coordinates = (49, 49)
            else:
                self.attack_coordinates = (12, 16)

        zerglings = self.get_units_by_type(obs, units.Zerg.Zergling)
        if len(zerglings) >= 10:
            if self.unit_type_is_selected(obs, units.Zerg.Zergling):
                if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                    return actions.FUNCTIONS.Attack_minimap("now",
                                                            self.attack_coordinates)

            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")

        spawning_pools = self.get_units_by_type(obs, units.Zerg.SpawningPool)
        if len(spawning_pools) == 0:
            if self.unit_type_is_selected(obs, units.Zerg.Drone):
                if self.can_do(obs, actions.FUNCTIONS.Build_SpawningPool_screen.id):
                    x = random.randint(0, 83)
                    y = random.randint(0, 83)

                    return actions.FUNCTIONS.Build_SpawningPool_screen("now", (x, y))

            drones = self.get_units_by_type(obs, units.Zerg.Drone)
            if len(drones) > 0:
                drone = random.choice(drones)

                return actions.FUNCTIONS.select_point("select_all_type", (drone.x,
                                                                          drone.y))
        if self.unit_type_is_selected(obs, units.Zerg.Larva):
            free_supply = (obs.observation.player.food_cap -
                           obs.observation.player.food_used)
            if free_supply == 0:
                if self.can_do(obs, actions.FUNCTIONS.Train_Overlord_quick.id):
                    return actions.FUNCTIONS.Train_Overlord_quick("now")

            if self.can_do(obs, actions.FUNCTIONS.Train_Zergling_quick.id):
                return actions.FUNCTIONS.Train_Zergling_quick("now")

        larvae = self.get_units_by_type(obs, units.Zerg.Larva)
        if len(larvae) > 0:
            larva = random.choice(larvae)

            return actions.FUNCTIONS.select_point("select_all_type", (larva.x,
                                                                      larva.y))

        return actions.FUNCTIONS.no_op()

def main(unused_argv):
    #agent = ZergBasicAgent()
    agent = TerranBasicAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                    #map_name="AbyssalReef",
                    map_name="Simple64",
                    #players=[sc2_env.Agent(sc2_env.Race.zerg),
                    players=[sc2_env.Agent(sc2_env.Race.terran),
                             sc2_env.Bot(sc2_env.Race.random,
                                         sc2_env.Difficulty.very_easy)],
                    agent_interface_format=features.AgentInterfaceFormat(
                      feature_dimensions=features.Dimensions(screen=84, minimap=64),
                      use_feature_units=True),
                    step_mul=1,
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
