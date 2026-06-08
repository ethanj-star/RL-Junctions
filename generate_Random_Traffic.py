import math
import random

# ==========================================
# 🚦 交通流核心超参数配置区
# ==========================================
SIM_DURATION = 3600  # 仿真总时长 (秒)
PEAK_TIME = 1800  # 高峰时刻 (秒)
STD_DEV = 600  # 高峰宽度

# 流量强度设定 (辆/小时)
MAIN_BASE, MAIN_PEAK = 400, 2000  # 东西主干道 (单向)
SIDE_BASE, SIDE_PEAK = 100, 600  # 南北支路 (单向)

RIGHT_TURN_PROB = 0.10  # 所有方向车辆的右转扰动概率 (10%)
OUTPUT_FILE = "traffic.random.rou.xml"


def get_p(t, base, peak):
    """计算第 t 秒发车的概率 p"""
    exponent = -0.5 * ((t - PEAK_TIME) / STD_DEV) ** 2
    flow = base + (peak - base) * math.exp(exponent)
    return flow / 3600.0


def generate_route_file():
    vehicles_xml = []
    veh_id = 0

    # 定义 8 条核心直行路线
    routes_straight = {
        "WE_MAIN": "left0A0 A0B0 B0C0 C0right0",  # 东西直行
        "EW_MAIN": "right0C0 C0B0 B0A0 A0left0",  # 西东直行
        "NS_A0": "top0A0 A0bottom0",  # A0 北->南
        "SN_A0": "bottom0A0 A0top0",  # A0 南->北
        "NS_B0": "top1B0 B0bottom1",  # B0 北->南
        "SN_B0": "bottom1B0 B0top1",  # B0 南->北
        "NS_C0": "top2C0 C0bottom2",  # C0 北->南
        "SN_C0": "bottom2C0 C0top2",  # C0 南->北
    }

    #  定义对应的 8 条右转路线 (严格匹配 SUMOroutes.net.xml)
    # 主干道车辆统一在中间的 B0 路口右转，支路车辆在各自路口右转
    routes_right = {
        "WE_MAIN": "left0A0 A0B0 B0bottom1",  # 东西向，在 B0 右转去南
        "EW_MAIN": "right0C0 C0B0 B0top1",  # 西东向，在 B0 右转去北
        "NS_A0": "top0A0 A0left0",  # A0 北向南，右转去西
        "SN_A0": "bottom0A0 A0B0",  # A0 南向北，右转去东
        "NS_B0": "top1B0 B0A0",  # B0 北向南，右转去西
        "SN_B0": "bottom1B0 B0C0",  # B0 南向北，右转去东
        "NS_C0": "top2C0 C0B0",  # C0 北向南，右转去西
        "SN_C0": "bottom2C0 C0right0",  # C0 南向北，右转去东
    }

    # 每一秒对所有流向进行独立泊松模拟
    for t in range(SIM_DURATION):
        for r_name in routes_straight.keys():
            # 根据主/支路属性选择概率基准
            base, peak = (MAIN_BASE, MAIN_PEAK) if "MAIN" in r_name else (SIDE_BASE, SIDE_PEAK)
            p = get_p(t, base, peak)

            if random.random() < p:
                # 掷骰子决定是直行还是右转
                if random.random() < RIGHT_TURN_PROB:
                    current_route = routes_right[r_name]
                    route_type = "right"
                else:
                    current_route = routes_straight[r_name]
                    route_type = "straight"

                # 写入 XML，加入 route_type 方便后续在 SUMO-GUI 中追踪观察
                xml_str = f'    <vehicle id="{r_name}_{route_type}_{veh_id}" type="human" depart="{t}.00">\n' \
                          f'        <route edges="{current_route}"/>\n' \
                          f'    </vehicle>'
                vehicles_xml.append(xml_str)
                veh_id += 1

    # 写入 XML
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n')
        f.write(
            '    <vType id="human" carFollowModel="Krauss" tau="1.0" accel="2.6" decel="4.5" length="5.0" color="white"/>\n\n')
        f.write('\n'.join(vehicles_xml))
        f.write('\n</routes>\n')

    print(f"全路网双向+右转随机流生成完毕，共计 {veh_id} 辆车。")


if __name__ == "__main__":
    generate_route_file()