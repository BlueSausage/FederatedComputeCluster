import math
import statistics

from agent import RZ
from typing import List


def generate_rz_list(num_rz: int,
                     costs: List[int],
                     sigma: List[int]) -> List[RZ]:
    rz_list = []
    for i in range(num_rz):
        rz = RZ(name=f'RZ{i+1}', mean_cost=costs[i], sigma=sigma[i])
        rz_list.append(rz)
    return rz_list


def determine_price(rz_list: List[RZ]) -> int:
    costs = []
    for rz in rz_list:
        costs.append(rz.cost)
    costs = sorted(costs)
    print(costs)
    return math.ceil(statistics.mean(costs))
    # return math.ceil(statistics.median(costs))


def run_episode():
    num_rz = 6
    costs = [5, 5, 5, 5, 5, 5]
    sigma = [2, 2, 2, 2, 2, 2]
    rz_list = generate_rz_list(num_rz=num_rz, costs=costs, sigma=sigma)
    price = determine_price(rz_list=rz_list)
    for rz in rz_list:
        earnings = max(0, price - rz.cost)
        print(
            f"{rz.name} hat Kosten von {rz.cost} und "
            f"einen Gewinn von {earnings}"
        )
    print()
    print(f"Preis für das bearbeiten eines Auftrags: {price}")
    print("----"*10)
