"""

Frenet optimal trajectory generator

author: Atsushi Sakai (@Atsushi_twi)

Ref:

- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame](https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)

- [Optimal trajectory generation for dynamic street scenarios in a Frenet Frame](https://www.youtube.com/watch?v=Cj6tAQe7UCY)

"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import cubic_spline_planner
import sys
import os
import pandas as pd
from scipy import sparse
import osqp


sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../QuinticPolynomialsPlanner/")

try:
    from quintic_polynomials_planner import QuinticPolynomial
except ImportError:
    raise

SIM_LOOP = 500

# Parameter
MAX_SPEED = 70.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 4.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 1.0  # maximum curvature [1/m]
MAX_ROAD_WIDTH = 7.0  # maximum road width [m]
D_ROAD_W = 1.0  # road width sampling length [m]
DT = 0.2  # time tick [s]
MAXT = 5.0  # max prediction time [m]
MINT = 4.0  # min prediction time [m]
TARGET_SPEED = 50.0 / 3.6  # target speed [m/s]
D_T_S = 5.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed
ROBOT_RADIUS = 2.0  # robot radius [m]

# cost weights
KJ = 0.1
KT = 0.1
KD = 1.0
KLAT = 1.0
KLON = 1.0

show_animation = True


class quartic_polynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, T):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * T ** 2, 4 * T ** 3],
                      [6 * T, 12 * T ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class Frenet_path:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []


def calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0):
    frenet_paths = []

    # generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):

        # Lateral motion planning
        for Ti in np.arange(MINT, MAXT, DT):
            fp = Frenet_path()

            # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Longitudinal motion planning (Velocity keeping)
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE, TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = quartic_polynomial(s0, c_speed, 0.0, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                tfp.cd = KJ * Jp + KT * Ti + KD * tfp.d[-1] ** 2
                tfp.cv = KJ * Js + KT * Ti + KD * ds
                tfp.cf = KLAT * tfp.cd + KLON * tfp.cv

                frenet_paths.append(tfp)

    return frenet_paths


def calc_global_paths(fplist, csp):
    for fp in fplist:

        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            iyaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(iyaw + math.pi / 2.0)
            fy = iy + di * math.sin(iyaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.hypot(dx, dy))

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

    return fplist


def check_collision(fp, ob):
    for i in range(len(ob[:, 0])):
        d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2)
             for (ix, iy) in zip(fp.x, fp.y)]

        collision = any([di <= ROBOT_RADIUS ** 2 for di in d])

        if collision:
            return False

    return True


def check_paths(fplist, ob):
    okind = []
    for i, _ in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            continue
        elif any([abs(a) > MAX_ACCEL for a in fplist[i].s_dd]):  # Max accel check
            continue
        elif any([abs(c) > MAX_CURVATURE for c in fplist[i].c]):  # Max curvature check
            continue
        elif not check_collision(fplist[i], ob):
            continue

        okind.append(i)

    return [fplist[i] for i in okind]


def frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob):
    fplist = calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0)
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist, ob)

    # find minimum cost path
    mincost = float("inf")
    bestpath = None
    for fp in fplist:
        if mincost >= fp.cf:
            mincost = fp.cf
            bestpath = fp

    return bestpath


def generate_target_course(x, y):
    csp = cubic_spline_planner.Spline2D(x, y)
    s = np.arange(0, csp.s[-1], 1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:  # 利用离散点x、y生成一条关于路程s的曲线？ 然后离散成x、y、yaw、curvature
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp


def generate_coarse_course(x, y):
    csp = cubic_spline_planner.Spline2D(x, y)
    s = np.arange(0, csp.s[-1], 1)  # 2m取一个间隔点，利用dp进行基于reference line的轨迹优化， 并保证一定的计算速度

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:  # 利用离散点x、y生成一条关于路程s的曲线？ 然后离散成x、y、yaw、curvature
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp


def project_to_cur(cur_x, cur_y, cur_theta, tar_x, tar_y):
    delta_x = tar_x - cur_x
    delta_y = tar_y - cur_y
    rel_x = delta_y * math.sin(cur_theta) + delta_x * math.cos(cur_theta)
    rel_y = delta_y * math.cos(cur_theta) - delta_x * math.sin(cur_theta)
    return rel_x, rel_y


class point:
    def __init__(self, x, y, parent=None, cost=10 ** 18, rtheta=0, length=1.0):
        self.x = x
        self.y = y
        self.cost = cost
        self.length = length
        self.tangent = [math.cos(rtheta), math.sin(rtheta)]
        self.parent = parent
        if parent == None:
            self.tangent = [math.cos(rtheta), math.sin(rtheta)]


def qp_shortest_path(wx, wy):

    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)  # 2m间隔的参考路径，可以计算每点的frenet坐标 法向量+切向量
    # tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)
    # tx_ori, ty_ori, tyaw_ori, tc_ori, csp_ori = generate_target_course(wx, wy)

    # osqp for nearest route
    total_dis = 0.0
    sampling_dis = 2.0
    sampling_num = int(csp.s[-1] // sampling_dis)
    E = np.zeros([2, sampling_num])
    H = np.zeros([sampling_num, sampling_num])
    B = np.zeros([1, sampling_num])
    next_pos = csp.calc_position(total_dis)
    pos_s = [next_pos]
    phi_1 = np.array(csp.calc_norm(total_dis))
    phi_s = [phi_1]
    for i in range(sampling_num-1):
        cur_pos = next_pos
        next_pos = csp.calc_position(total_dis + sampling_dis)
        pos_s.append(next_pos)
        delta_x = next_pos[0] - cur_pos[0]
        delta_y = next_pos[1] - cur_pos[1]
        E_i = np.zeros([2, sampling_num])
        phi = phi_1
        phi_1 = np.array(csp.calc_norm(total_dis + sampling_dis))
        phi_s.append(phi_1)
        phi_xi = np.array([[phi_1[0], -phi[0]]])
        phi_yi = np.array([[phi_1[1], -phi[1]]])
        E_i[0][i] = 1
        E_i[1][i+1] = 1
        B_i = 2 * delta_x * np.dot(phi_xi, E_i) + 2 * delta_y * np.dot(phi_yi, E_i)
        H_si_t = np.dot(phi_xi.T, phi_xi) + np.dot(phi_yi.T, phi_yi)
        H += np.dot(np.dot(E_i.T, H_si_t), E_i)
        B += B_i
        total_dis += sampling_dis
    P = sparse.csc_matrix(H)
    q = B[0]
    A = sparse.csc_matrix(np.eye(sampling_num))
    l = np.array([-2.5 for i in range(sampling_num)])
    l[0] = -0.1
    l[-1] = -0.1
    u = np.array([2.5 for i in range(sampling_num)])
    u[0] = 0.1
    u[-1] = 0.1
    prob = osqp.OSQP()
    # Setup workspace and change alpha parameter
    prob.setup(P, q, A, l, u, alpha=1.0)
    # Solve problem
    res = prob.solve()
    print(res.x)
    shortest_x = []
    shortest_y = []
    for i in range(sampling_num):
        shortest_x.append(pos_s[i][0] + res.x[i] * phi_s[i][0])
        shortest_y.append(pos_s[i][1] + res.x[i] * phi_s[i][1])
    tx_shortest, ty_shortest, tyaw_shortest, tc_shortest, csp_shortest = generate_target_course(shortest_x, shortest_y)
    return tx_shortest, ty_shortest
    #plt.plot(tx_shortest, ty_shortest, '-r', tx_ori, ty_ori, '-b')
    #plt.show()

def main():
    print(__file__ + " start!!")

    data = pd.read_csv(r"..\软件大赛——极速赛道规划控制\saic_2020.csv")
    x_data = data['x']
    y_data = data['y']
    # way points
    wx = x_data
    wy = y_data
    # obstacle lists
    ob = np.array([[90.0934, 710.9762],
                   [130.0934, 707.9762],
                   ])
    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)  # 2m间隔的参考路径，可以计算每点的frenet坐标 法向量+切向量
    tx_coarse, ty_coarse, tyaw_coarse, tc_coarse, csp_coarse = generate_coarse_course(wx, wy)
    boundary_left_x = []
    boundary_left_y = []
    boundary_right_x = []
    boundary_right_y = []
    for i in range(len(csp_coarse.s)-1):
        temp_pos = csp_coarse.calc_position(csp_coarse.s[i])
        temp_norm = csp_coarse.calc_norm(csp_coarse.s[i])
        boundary_left_x.append(temp_pos[0]+3*temp_norm[0]), boundary_left_y.append(temp_pos[1]+3*temp_norm[1])
        boundary_right_x.append(temp_pos[0] - 3 * temp_norm[0]), boundary_right_y.append(temp_pos[1] - 3 * temp_norm[1])
    tx_left, ty_left, tyaw_left, tc_left, csp_left = generate_target_course(boundary_left_x, boundary_left_y)
    tx_right, ty_right, tyaw_right, tc_right, csp_right = generate_target_course(boundary_right_x, boundary_right_y)
    # tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)
    tx_ori, ty_ori, tyaw_ori, tc_ori, csp_ori = generate_target_course(wx, wy)


    # minimum curvature planning
    total_dis_c = 0
    sampling_dis_c = 2.0
    sampling_num_c = int(csp.s[-1] // sampling_dis_c)
    D = np.zeros([sampling_num_c-2, sampling_num_c])
    di_x = np.zeros([sampling_num_c, sampling_num_c])
    di_y = np.zeros([sampling_num_c, sampling_num_c])
    x_r = np.zeros([sampling_num_c, 1])
    y_r =np.zeros([sampling_num_c, 1])
    for i in range(sampling_num_c-2):
        D[i][i] = D[i][i+2] = 1.0
        D[i][i+1] = -2.0
    pos_sc = []
    phi_sc = []
    for i in range(sampling_num_c):
        phi_c = csp.calc_norm(total_dis_c)
        phi_sc.append(phi_c)
        pos_c_x, pos_c_y = csp.calc_position(total_dis_c)
        pos_sc.append([pos_c_x, pos_c_y])
        di_x[i][i] = phi_c[0]
        di_y[i][i] = phi_c[1]
        x_r[i][0] = pos_c_x
        y_r[i][0] = pos_c_y
        total_dis_c += sampling_dis_c
    D *= 0.5
    H_T = 2.0 * np.dot(np.dot(np.dot(di_x.T, D.T), D), di_x) + np.dot(np.dot(np.dot(di_y.T, D.T), D), di_y)
    B_T = 2 * np.dot(np.dot(np.dot(x_r.T, D.T), D), di_x) + 2 * np.dot(np.dot(np.dot(y_r.T, D.T), D), di_y)
    P_c = sparse.csc_matrix(H_T)
    q_c = B_T[0]
    A_c = sparse.csc_matrix(np.eye(sampling_num_c))
    l_c = np.array([-2.5 for i in range(sampling_num_c)])
    l_c[0] = -0.1
    l_c[-1] = -0.1
    u_c = np.array([2.5 for i in range(sampling_num_c)])
    u_c[0] = 0.1
    u_c[-1] = 0.1
    prob_c = osqp.OSQP()
    # Setup workspace and change alpha parameter
    prob_c.setup(P_c, q_c, A_c, l_c, u_c, alpha=1.0)
    # Solve problem
    res = prob_c.solve()
    print(res.x)
    smooth_x_c = []
    smooth_y_c = []
    for i in range(sampling_num_c):
        smooth_x_c.append(pos_sc[i][0] + res.x[i] * phi_sc[i][0])
        smooth_y_c.append(pos_sc[i][1] + res.x[i] * phi_sc[i][1])
    tx_c, ty_c, tyaw_c, tc_c, csp_c = generate_target_course(smooth_x_c, smooth_y_c)
    tx_shortest, ty_shortest = qp_shortest_path(wx, wy)
    #labels = ['max_cur', 'reference line', 'shortest']
    plt.plot(tx_c, ty_c, '-r', label="max_cur")
    plt.plot(tx_ori, ty_ori, '-b', label="reference")
    plt.plot(tx_shortest, ty_shortest, '-g', label="shortest")
    plt.plot(tx_left, ty_left, ':k')
    plt.plot(tx_right, ty_right, ':k')
    plt.legend()
    plt.show()
    # dp method
    lateral_cand = [0.0]

    for x in np.linspace(-2.5, 2.5, 40):
        lateral_cand.append(x)
    """lateral_cand = [0, -0.25, 0.25, -0.5, 0.5, -0.75, 0.75,
                    -1, 1, -1.25, 1.25, 1.5, -1.5, -1.75, 1.75,
                    -2, 2, -2.25, 2.25, -2.5, 2.5 ]  # lateral distance candidate"""

    initial_norm = csp.calc_norm(0.0)
    dp = [[point(tx[0] + x * initial_norm[0], ty[0] + x * initial_norm[1], cost=0, rtheta=math.pi/2) for x in lateral_cand]]
    dp[0][0] = point(tx[0], ty[0], cost=0, rtheta=tyaw[0])
    delta_s = 3.0  # sampling density
    total_s = delta_s
    w_coarse = 0.999
    while total_s < csp.s[-1]:
        norm_vec = csp.calc_norm(total_s)
        ref_x, ref_y = csp.calc_position(total_s)
        temp = []
        for ind, lat_offset in enumerate(lateral_cand):
            temp_x = ref_x + norm_vec[0] * lat_offset
            temp_y = ref_y + norm_vec[1] * lat_offset
            cur_cost = 10 ** 18
            cur_route = point(temp_x, temp_y)
            for ind_2 in range(len(lateral_cand)):
                # calc cost from parent node to current node
                par_x = dp[-1][ind_2].x
                par_y = dp[-1][ind_2].y
                dis = np.hypot(temp_x-par_x, temp_y-par_y)
                tangent = [(temp_x-par_x)/dis, (temp_y-par_y)/dis]
                cur_theta = math.atan2(tangent[1], tangent[0])
                k = [(tangent[0]-dp[-1][ind_2].tangent[0])/dis, (tangent[1]-dp[-1][ind_2].tangent[1])/dis]
                #np.hypot(k[0], k[1])
                acosvalue = tangent[0] * dp[-1][ind_2].tangent[0] + tangent[1] * dp[-1][ind_2].tangent[1]


                if abs(acosvalue) > 1.0:
                    acosvalue = 1.0 * np.sign(acosvalue)
                # temp_cost = w_coarse * math.acos(acosvalue) / (dp[-1][ind_2].length + dis) + dp[-1][ind_2].cost + dis * (1-w_coarse)#
                temp_cost = dp[-1][ind_2].cost + w_coarse * np.hypot(k[0], k[1])# dis  # * (1 - w_coarse)# 最大曲率
                # temp_cost = dp[-1][ind_2].cost + dis #* (1 - w_coarse)# w_coarse * np.hypot(k[0], k[1]) +  # 最短路径

                for i, pos in enumerate(ob):
                    pro_x, pro_y = project_to_cur(temp_x, temp_y, cur_theta, pos[0], pos[1])
                    if -3.0 < pro_x < 6.0 and abs(pro_y) < 3.0:
                        temp_cost += 10**18         # 避障惩罚项
                if temp_cost < cur_cost:
                    cur_cost = temp_cost
                    cur_route.length = dis
                    cur_route.tangent = tangent
                    cur_route.cost = temp_cost
                    cur_route.parent = ind_2
            temp.append(cur_route)
        dp.append(temp)
        total_s += delta_s

    x_points = []
    y_points = []
    cur_node = min(dp[-1], key=lambda x:x.cost)
    cur_node_n = -1
    while cur_node.parent is not None:
        x_points.append(cur_node.x)
        y_points.append(cur_node.y)
        cur_node_n -= 1
        cur_node = dp[cur_node_n][cur_node.parent]
    x_points.append(cur_node.x)
    y_points.append(cur_node.y)
    x_points.reverse()
    y_points.reverse()
    tx_opt, ty_opt, tyaw_opt, tc_opt, csp_opt = generate_coarse_course(x_points, y_points)
    plt.plot(tx_opt, ty_opt, 'r-', tx_ori, ty_ori, 'b-')
    plt.show()

    # initial state
    c_speed = 40.0 / 3.6  # current speed [m/s]
    c_d = 0.0  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current latral acceleration [m/s]
    s0 = 0.0  # current course position

    area = 50.0  # animation area length [m]

    for i in range(SIM_LOOP):
        path = frenet_optimal_planning(
            csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob)

        s0 = path.s[1]
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
        c_speed = path.s_d[1]

        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(tx, ty)
            plt.plot(ob[:, 0], ob[:, 1], "xk")
            plt.plot(path.x[1:], path.y[1:], "-or")
            plt.plot(path.x[1], path.y[1], "vc")
            plt.xlim(path.x[1] - area, path.x[1] + area)
            plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.title("v[km/h]:" + str(c_speed * 3.6)[0:4])
            plt.grid(True)
            plt.pause(0.0001)

    print("Finish")
    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    main()
