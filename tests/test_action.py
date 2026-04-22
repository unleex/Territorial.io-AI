from custom_environment.custom_environment_v0 import CustomEnvironment

env = CustomEnvironment()
env.step({0: [0, 10]})
assert env.game.id_to_country[0].size > 0
print("Passed")
