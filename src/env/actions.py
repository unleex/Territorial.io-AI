from custom_environment.env.countryClass import country


class Action: ...


class Attack(Action):
    def __init__(self, attacker: country, attacked: country):
        self.attacker = attacker
        self.attacked = attacked
