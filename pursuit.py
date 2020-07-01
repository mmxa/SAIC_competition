import numpy as np
import math
import matplotlib.pyplot as plt

"""————————————————
版权声明：本文为CSDN博主「AdamShan」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/AdamShan/article/details/80555174

commit:2020.6.26
add anchor distance from:Kuwata-2009-Real-Time Motion Pla
"""

k = 0.1  # 前视距离系数
Lfc = 2.0  # 前视距离
Kp = 1.0  # 速度P控制器系数
dt = 0.1  # 时间间隔，单位：s
L = 2.9  # 车辆轴距，单位：m
anchor_distance = 0.2   # anchor point relative to rear axel

class VehicleState:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v


def update(state, a, delta):
    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
    state.v = state.v + a * dt

    return state


def PControl(target, current):
    a = Kp * (target - current)

    return a


def pure_pursuit_control(state, cx, cy, pind):
    ind = calc_target_index(state, cx, cy)

    if pind >= ind:
        ind = pind

    if ind < len(cx):
        tx = cx[ind]
        ty = cy[ind]
    else:
        tx = cx[-1]
        ty = cy[-1]
        ind = len(cx) - 1

    # Add a concept of anchor point:
    anchor_x = state.x + anchor_distance * math.cos(state.yaw)
    anchor_y = state.y + anchor_distance * math.sin(state.yaw)
    alpha = math.atan2(ty - anchor_y, tx - anchor_x) - state.yaw

    #alpha = math.atan2(ty - state.y, tx - state.x) - state.yaw

    if state.v < 0:  # back
        alpha = math.pi - alpha
    # 为什么LF距离只能通过第一个公式计算，这个结果与实际路径点相对于车辆当前位置具有较大误差？？？？
    # Lf = abs(k * state.v) + Lfc
    Lf = math.hypot(ty - anchor_y, tx - anchor_x)
    delta = math.atan2(2.0 * L * math.sin(alpha) / Lf, 1.0)

    return delta, ind


def calc_target_index(state, cx, cy):
    # 搜索最临近的路点， 找到距离当前state距离最近的路径点
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]
    d = [abs(math.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]
    ind = d.index(min(d))
    L = 0.0

    Lf = abs(k * state.v) + Lfc

    while Lf > L and (ind + 1) < len(cx):   # 找到从最近路径点出发距离当前state前向距离大于预设距离的路径点
        dx = cx[ind + 1] - cx[ind]
        dy = cx[ind + 1] - cx[ind]
        L += math.sqrt(dx ** 2 + dy ** 2)
        ind += 1

    return ind


def main():
    #  设置目标路点ix
    cx = np.arange(10, 50, 1)
    cy = [math.sin(ix / 5.0) * ix / 2.0 for ix in cx]
    #cy = [0.01*ix**3-0.02*ix**2 / 2.0 for ix in cx]
    target_speed = 10.0 / 3.6  # [m/s]

    T = 100.0  # 最大模拟时间

    # 设置车辆的初始状态
    state = VehicleState(x=-0.0, y=-3.0, yaw=2.0, v=0.0)

    lastIndex = len(cx) - 1
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    delta = [0.0]
    target_ind = calc_target_index(state, cx, cy)

    while T >= time and lastIndex > target_ind:
        ai = PControl(target_speed, state.v)
        di, target_ind = pure_pursuit_control(state, cx, cy, target_ind)
        state = update(state, ai, di)

        time = time + dt

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        delta.append(di)

        plt.cla()
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(x, y, "-b", label="trajectory")
        plt.plot(cx[target_ind], cy[target_ind], "go", label="target")
        plt.axis("equal")
        plt.grid(True)
        plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
        plt.pause(0.001)

    """plt.cla()
    plt.grid(True)
    plt.plot(t, delta, '-b', label="delta")"""

if __name__ == '__main__':
    main()
    plt.show()
