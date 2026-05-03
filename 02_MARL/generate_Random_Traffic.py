import math
import random


# 交通流核心超参数配置区 — core hyperparameter configuration area for traffic flow

SIM_DURATION = 3600  # 仿真总时长 (秒) — total simulation duration (seconds)
PEAK_TIME = 1800  # 高峰时刻 (秒) — peak time (seconds)
STD_DEV = 600  # 高峰宽度 — peak width (standard deviation)

# 流量强度设定 (辆/小时) — flow intensity settings (vehicles/hour)
MAIN_BASE, MAIN_PEAK = 400, 2000  # 东西主干道 (单向) — East-West main road (one-way)
SIDE_BASE, SIDE_PEAK = 100, 600  # 南北支路 (单向) — North-South side road (one-way)

RIGHT_TURN_PROB = 0.10  # 所有方向车辆的右转扰动概率 (10%) — right turn perturbation probability for all directions (10%)
OUTPUT_FILE = "traffic.random.rou.xml"


def get_p(t, base, peak):
    """计算第 t 秒发车的概率 p — calculate the probability p of dispatching a vehicle at second t"""
    exponent = -0.5 * ((t - PEAK_TIME) / STD_DEV) ** 2
    flow = base + (peak - base) * math.exp(exponent)
    return flow / 3600.0  #flow: cars/hour  return cars/second


def generate_route_file():
    vehicles_xml = []
    veh_id = 0

    # 定义 8 条核心直行路线 — define 8 core straight routes
    routes_straight = {
        "WE_MAIN": "left0A0 A0B0 B0C0 C0right0",  # 东西直行 — West-East straight
        "EW_MAIN": "right0C0 C0B0 B0A0 A0left0",  # 西东直行 — East-West straight
        "NS_A0": "top0A0 A0bottom0",  # A0 北->南 — A0 North->South
        "SN_A0": "bottom0A0 A0top0",  # A0 南->北 — A0 South->North
        "NS_B0": "top1B0 B0bottom1",  # B0 北->南 — B0 North->South
        "SN_B0": "bottom1B0 B0top1",  # B0 南->北 — B0 South->North
        "NS_C0": "top2C0 C0bottom2",  # C0 北->南 — C0 North->South
        "SN_C0": "bottom2C0 C0top2",  # C0 南->北 — C0 South->North
    }

    # 定义对应的 8 条右转路线 (严格匹配 SUMOroutes.net.xml) — define corresponding 8 right-turn routes (strictly matches SUMOroutes.net.xml)
    # 主干道车辆统一在中间的 B0 路口右转，支路车辆在各自路口右转 — main road vehicles uniformly turn right at the middle B0 intersection, side road vehicles turn right at their respective intersections
    routes_right = {
        "WE_MAIN": "left0A0 A0B0 B0bottom1",  # 东西向，在 B0 右转去南 — West-East, turn right at B0 to South
        "EW_MAIN": "right0C0 C0B0 B0top1",  # 西东向，在 B0 右转去北 — East-West, turn right at B0 to North
        "NS_A0": "top0A0 A0left0",  # A0 北向南，右转去西 — A0 North to South, turn right to West
        "SN_A0": "bottom0A0 A0B0",  # A0 南向北，右转去东 — A0 South to North, turn right to East
        "NS_B0": "top1B0 B0A0",  # B0 北向南，右转去西 — B0 North to South, turn right to West
        "SN_B0": "bottom1B0 B0C0",  # B0 南向北，右转去东 — B0 South to North, turn right to East
        "NS_C0": "top2C0 C0B0",  # C0 北向南，右转去西 — C0 North to South, turn right to West
        "SN_C0": "bottom2C0 C0right0",  # C0 南向北，右转去东 — C0 South to North, turn right to East
    }

    # 每一秒对所有流向进行独立泊松模拟 — independent Poisson simulation for all flows every second
    for t in range(SIM_DURATION):
        for r_name in routes_straight.keys():
            # 根据主/支路属性选择概率基准 — select probability baseline based on main/side road attributes
            base, peak = (MAIN_BASE, MAIN_PEAK) if "MAIN" in r_name else (SIDE_BASE, SIDE_PEAK)
            #传入高斯概率
            p = get_p(t, base, peak)

            #如果随机数小于概率，则生成一个车，不然就不生成。随着p越来越大，宏观上车辆就越来越多，因为概率越来越高。但是微观上还是靠下面的代码来随机生成。
            if random.random() < p:
                # 掷骰子决定是直行还是右转 — random decide whether to go straight or turn right
                if random.random() < RIGHT_TURN_PROB:
                    current_route = routes_right[r_name]
                    route_type = "right"
                else:
                    current_route = routes_straight[r_name]
                    route_type = "straight"

                # 写入 XML，加入 route_type 方便后续在 SUMO中追踪观察
                # write to XML, add route_type for easy tracking and observation later in SUMO
                xml_str = f'    <vehicle id="{r_name}_{route_type}_{veh_id}" type="human" depart="{t}.00">\n' \
                          f'        <route edges="{current_route}"/>\n' \
                          f'    </vehicle>'
                vehicles_xml.append(xml_str)
                veh_id += 1

    # 写入 XML — write to XML
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n')
        f.write(
            '    <vType id="human" carFollowModel="Krauss" tau="1.0" accel="2.6" decel="4.5" length="5.0" color="white"/>\n\n')
        f.write('\n'.join(vehicles_xml))
        f.write('\n</routes>\n')

    print(f"全路网双向+右转随机流生成完毕，共计 {veh_id} 辆车。")


if __name__ == "__main__":
    generate_route_file()