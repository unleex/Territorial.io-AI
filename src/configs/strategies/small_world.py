from configs.strategies.common import GAMMA_DECAY

N_PLAYERS = 8
MAP_SIZE = (80, 80)
N_TRAINABLES = 1
LEAGUE_UPDATE_PLACE_PERCENTILE = 0.3
BOT_EXPANSION_BOOST = 1
GAME_MAX_TURNS = 1000
RUN_NAME = "scalar_commit"
TERMINAL_REWARD_COEFF = 1
POLICY_COLORS = {
    # Main Policy
    "p0": "#FFFFFF",
    # Frozen Policies (Shades of Blue, Purple, and Pink)
    "p0_v0": "#0D47A1",
    "p0_v1": "#4A148C",
    "p0_v2": "#D500F9",
    "p0_v3": "#AA00FF",
    "p0_v4": "#2196F3",
    "p0_v5": "#42A5F5",
    "p0_v6": "#64B5F6",
    "p0_v7": "#90CAF9",
    "p0_v8": "#2962FF",
    "p0_v9": "#2979FF",
    "p0_v10": "#448AFF",
    "p0_v12": "#6A1B9A",
    "p0_v13": "#7B1FA2",
    "p0_v14": "#8E24AA",
    "p0_v15": "#9C27B0",
    "p0_v16": "#AB47BC",
    "p0_v17": "#BA68C8",
    "p0_v19": "#E040FB",
    "p0_v21": "#880E4F",
    "p0_v22": "#AD1457",
    "p0_v23": "#C2185B",
    "p0_v24": "#D81B60",
    "p0_v25": "#E91E63",
    "p0_v26": "#EC407A",
    "p0_v27": "#F06292",
    "p0_v28": "#F48FB1",
    "p0_v29": "#FF178B",
    "p0_v30": "#F50057",
    "p0_v31": "#FF4081",
    # Bots (Shades of Green, Orange, and Yellow)
    "bot0": "#1B5E20",
    "bot1": "#81C784",
    "bot2": "#00C853",
    "bot3": "#69F0AE",
    "bot4": "#AEEA00",
    "bot5": "#C6FF00",
    "bot6": "#FFF176",
    "bot7": "#E65100",
}
