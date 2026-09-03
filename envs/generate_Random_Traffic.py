import math
import random
from pathlib import Path

# 这些参数定义随机车流的仿真时长、高峰时刻和主路/支路交通强度。
# These parameters define the random traffic duration, peak time, and main-road/side-road flow intensity.
SIM_DURATION = 3600
PEAK_TIME = 1800
STD_DEV = 600
MAIN_BASE, MAIN_PEAK = 400, 2000
SIDE_BASE, SIDE_PEAK = 100, 600
RIGHT_TURN_PROB = 0.10
OUTPUT_FILE = Path(__file__).resolve().with_name("traffic.random.rou.xml")


# 根据高斯形状的高峰需求计算每秒发车概率。
# This calculates the per-second departure probability from a Gaussian-shaped peak demand profile.
def get_p(t, base, peak):
    exponent = -0.5 * ((t - PEAK_TIME) / STD_DEV) ** 2
    flow = base + (peak - base) * math.exp(exponent)
    return flow / 3600.0


# 生成器可接收独立 seed 和输出路径，使并行环境得到可复现且互不覆盖的车流文件。
# An explicit seed and output path give parallel environments reproducible, isolated route files.
def generate_route_file(seed=None, output_file=OUTPUT_FILE):
    rng = random.Random(seed) if seed is not None else random
    output_path = Path(output_file)
    vehicles_xml = []
    veh_id = 0

    routes_straight = {
        "WE_MAIN": "left0A0 A0B0 B0C0 C0right0",
        "EW_MAIN": "right0C0 C0B0 B0A0 A0left0",
        "NS_A0": "top0A0 A0bottom0",
        "SN_A0": "bottom0A0 A0top0",
        "NS_B0": "top1B0 B0bottom1",
        "SN_B0": "bottom1B0 B0top1",
        "NS_C0": "top2C0 C0bottom2",
        "SN_C0": "bottom2C0 C0top2",
    }
    routes_right = {
        "WE_MAIN": [
            "left0A0 A0bottom0",
            "left0A0 A0B0 B0bottom1",
            "left0A0 A0B0 B0C0 C0bottom2",
        ],
        "EW_MAIN": [
            "right0C0 C0top2",
            "right0C0 C0B0 B0top1",
            "right0C0 C0B0 B0A0 A0top0",
        ],
        "NS_A0": ["top0A0 A0left0"],
        "SN_A0": ["bottom0A0 A0B0"],
        "NS_B0": ["top1B0 B0A0"],
        "SN_B0": ["bottom1B0 B0C0"],
        "NS_C0": ["top2C0 C0B0"],
        "SN_C0": ["bottom2C0 C0right0"],
    }

    # 每一秒独立抽样一次车辆生成事件，主路和支路使用不同流量参数。
    # A vehicle generation event is sampled independently each second, with different flow settings for main and side roads.
    for t in range(SIM_DURATION):
        for route_name, straight_route in routes_straight.items():
            base, peak = (MAIN_BASE, MAIN_PEAK) if "MAIN" in route_name else (SIDE_BASE, SIDE_PEAK)
            if rng.random() >= get_p(t, base, peak):
                continue

            if rng.random() < RIGHT_TURN_PROB:
                current_route = rng.choice(routes_right[route_name])
                route_type = "right"
            else:
                current_route = straight_route
                route_type = "straight"

            vehicles_xml.append(
                f'    <vehicle id="{route_name}_{route_type}_{veh_id}" type="human" depart="{t}.00">\n'
                f'        <route edges="{current_route}"/>\n'
                f'    </vehicle>'
            )
            veh_id += 1

    # 默认仍写入共享 envs；训练可传入当前 run 的独立路径。
    # The shared envs file remains the default, while training may provide a run-local path.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n')
        f.write('    <vType id="human" carFollowModel="Krauss" tau="1.0" accel="2.6" decel="4.5" length="5.0" color="white"/>\n\n')
        f.write("\n".join(vehicles_xml))
        f.write("\n</routes>\n")

    print(f"Random traffic route file generated: {output_path} ({veh_id} vehicles, seed={seed})")
    return output_path


if __name__ == "__main__":
    generate_route_file()
