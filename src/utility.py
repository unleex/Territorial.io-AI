# used in environment and in bot.py to mimic NN agent's action space
def permute_id(id_permutation, original_id: int) -> int:
    return int(id_permutation[original_id + 1])
