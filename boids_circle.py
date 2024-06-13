# 基于BOIDS的阿式圆围捕法
import time
import argparse
import numpy as np
import random
from random import uniform

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from elements.assets  import *

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES =5
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_VISION = False
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = True
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 100
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DISTANCE=0.7
# C1_alpha =3
C2_alpha =25
C1_gamma =0
C2_gamma =25
def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        vision=DEFAULT_VISION,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        aggregate=DEFAULT_AGGREGATE,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    #### Initialize the simulation #############################
    '''
    该场景中设置了四种追逐无人机，与一架逃跑无人机
    '''
    H = .5
    R =1
    P=np.array([[1,0,H+0.5],[0,1,H+0.5],[-1,0,H+0.5],[0,-1,H+0.5]]) #追逐无人机
    E=np.array([[-5,0,H]])  #逃跑无人机
    VP=0.3
    VE=0.2#阿波罗尼奥斯圆速度需满足条件
    delt=VP/VE  #阿波罗尼奥斯圆需满足一定条件
    INIT_XYZS= np.vstack( (P, E) )
    INIT_RPYS = np.array([[0, 0,  0] for i in range(num_drones)])
    AGGR_PHY_STEPS = int(simulation_freq_hz/control_freq_hz) if aggregate else 1
    env = CtrlAviary(drone_model=drone,
                     num_drones=num_drones,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=physics,
                     neighbourhood_radius=0.7,
                     freq=simulation_freq_hz,
                     aggregate_phy_steps=AGGR_PHY_STEPS,
                     gui=gui,
                     record=record_video,
                     obstacles=obstacles,
                     user_debug_gui=user_debug_gui
                     )
    logger = Logger(logging_freq_hz=int(simulation_freq_hz / AGGR_PHY_STEPS),
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    elif drone in [DroneModel.HB]:
        ctrl = [SimplePIDControl(drone_model=drone) for i in range(num_drones - 1)]

    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / control_freq_hz)) #控制间隔与仿真间隔分开
    action = {str(i): np.array([0, 0, 0, 0]) for i in range(num_drones)} #不论是在用路径规划的方法或是采用控制无人机速度的方法最终作用于发电机电机转速
    START = time.time()

    for x in range(0, int(duration_sec * env.SIM_FREQ), AGGR_PHY_STEPS):


        obs, reward, done, info = env.step(action)  #在obs中存储了所有无人机的状态与邻居信息（state,neighbors）
        u = np.zeros((num_drones, 2))   #定义一个控制变量，该变量用于速度控制调整

        P_pos=dict() #用字典存储每个智能体的位置信息

        if x % CTRL_EVERY_N_STEPS == 0:
            for j in range(num_drones-1):
                P_pos[j] = obs[str(j)]["state"][:2] #追捕无人机的位置信息
            E_pos=obs[str(num_drones-1)]["state"][:2]# 逃跑无人机的位置信息

        ##根据无人机的位置信息确定圆心，半径等,并设置其他参数
            O=dict()
            r=dict()
            d=dict()
            v=np.zeros((num_drones-1, 2))
            u = np.zeros((num_drones-1, 2))
            p = list(P_pos.values())
            np1 = np.vstack(p)
            maxxy = np.max(np1, axis=0)
            minxy = np.min(np1, axis=0)
            maxx = maxxy[0]
            maxy = maxxy[1]
            minx = minxy[0]
            miny = minxy[1]
            for i in range(num_drones-1):
                agent_p = P_pos[i][0:2]
                # 获取所有agent位置信息

                agent_q = obs[str(i)]["state"][10:12]  # 获取所有agent的速度信息
                neighbor_idxs = obs[str(i)]["neighbors"]  # 获取每个agent的邻居信息，该数值需要提前指出
                neighbor_idxs[num_drones - 1] = 0
                neighbors_p = []  # 存储邻居的位置信息
                neighbors_q = []  # 存储邻居的速度信息
             # 判断是否需要避障，获得避障参数
                if sum(neighbor_idxs) > 1:
                    ##获取每个邻居的状态信息，该部分代码需要优化
                    a = (np.where(neighbor_idxs == 1.0))

                    # neighbors_p = [obs[((np.where(neighbor_idxs==1.0)[0]))]["state"][:2]]
                    # neighbors_q = [obs[((np.where(neighbor_idxs==1.0)[0]))][][10:12]]
                    for j in a:
                        for t in j:
                            l = obs[str(t)]["state"][:2]
                            neighbors_p.append(l)
                            neighbors_q.append(obs[str(t)]["state"][10:12])
                    neighbors_p = np.array(neighbors_p)
                    neighbors_q = np.array(neighbors_q)

                    n_ij = get_n_ij(agent_p, neighbors_p)  # 利用当前智能体位置信息以及其他邻居agent位置信息计算

                    term1 = C2_alpha * np.sum(phi_alpha(sigma_norm(neighbors_p - agent_p)) * n_ij, axis=0)
                    # term2
                    a_ij = get_a_ij(agent_p, neighbors_p)
                    term2 = C2_alpha * np.sum(a_ij * (neighbors_q - agent_q), axis=0)

                    # u_alpha
                    u_alpha = term1 + term2  # 排斥力   u_alpa用于队内避障
                else:
                    u_alpha=0

              ####根据策略获得无人机行动速度，集群目的地
                if (E_pos[0] > (minx + 0.2)) & (E_pos[1] > (miny + 0.2)) & (E_pos[0] < (maxx - 0.2)) & (
                        E_pos[1] < (maxy - 0.2)):
                    # 这种情况下需要根据阿波罗圆进行策略，先获取相应的信息，上面不需要，避免重复运算
                    for j in range(num_drones - 1):
                        O[j] = np.array((P_pos[j][0] - delt ** 2 * (E_pos[0])) / (1 - delt ** 2),
                                        (P_pos[j][1] - delt ** 2 * (E_pos[1])) / (1 - delt ** 2))  # 圆心
                        r[j] = delt * np.sqrt((P_pos[j][0] - E_pos[0]) ** 2 + (P_pos[j][0] - E_pos[0]) ** 2) / (
                                1 - delt ** 2)  # 半径
                        d[j] = np.linalg.norm(O[j] - E_pos)  # 与逃跑无人机的距离

                    index = max(d, key=lambda key: d[key])  # 确定哪架无人机距离逃跑无人机最远



                    if np.linalg.norm(O[i] - (O[(i + 1) % (num_drones - 1)])) > r[i] + r[
                        (i + 1) % (num_drones - 1)]:
                        v[i] = (obs[str((i + 1) % (num_drones - 1))]["state"][:2] - obs[str(i)]["state"][:2]) / \
                               (np.linalg.norm(
                                   obs[str((i + 1) % (num_drones - 1))]["state"][:2] - obs[str(i)]["state"][:2]))
                    # if np.linalg.norm(O[i] - (O[(i -1) % (num_drones - 1)])) > r[i] + r[
                    #     (i -1) % (num_drones - 1)]:
                    #     v[i] = (obs[str((i -1) % (num_drones - 1))]["state"][:2] - obs[str(i)]["state"][:2]) / \
                    #            (np.linalg.norm(
                    #                obs[str((i -1) % (num_drones - 1))]["state"][:2] - obs[str(i)]["state"][:2]))

                    # print(" %d 正在包围" %(i))

                    else:
                        if i == index:
                            v[i] = (E_pos - obs[str(i)]["state"][:2]) / np.linalg.norm(E_pos - obs[str(i)]["state"][:2])
                        else:
                            v[i] = (E_pos - obs[str(i)]["state"][:2]) / np.linalg.norm(E_pos - obs[str(i)]["state"][:2])



                    u_gamma = -C1_gamma * sigma_1(agent_p - np.array([E_pos[0], E_pos[1]])) - C2_gamma * (
                            agent_q - np.array([[v[i][0], v[i][1]]]))  # 吸引力，利用目标点或者速度来设定
                    u[i] = u_alpha + u_gamma
                    v1 = agent_q
                    v_now = v1 + (np.array([u[i][0], u[i][1]]) * (1 / 48))
                    v_now = v_now / np.linalg.norm(v_now)

                    action[str(i)], _, _ = ctrl[i].computeControl(
                        control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                        cur_pos=obs[str(i)]["state"][0:3],
                        cur_quat=obs[str(i)]["state"][3:7],
                        cur_vel=obs[str(i)]["state"][10:13],
                        cur_ang_vel=obs[str(i)]["state"][13:16],
                        target_pos=obs[str(i)]["state"][0:3],
                        # same as the current position
                        target_rpy=np.array([0, 0, obs[str(i)]["state"][9]]),
                        # keep current yaw
                        target_vel=0.03 * env.MAX_SPEED_KMH * (
                                1000 / 3600) * np.abs(VP) * np.array([v_now[0], v_now[1], 0]))
                else:

                    v[i] = (E_pos[0:2] - agent_p) / np.linalg.norm(E_pos[0:2] - agent_p)
                    u_gamma = -C1_gamma * sigma_1(agent_p - np.array([E_pos[0], E_pos[1]])) - C2_gamma * (
                            agent_q - np.array([[v[i][0], v[i][1]]]))  # 吸引力，利用目标点或者速度来设定
                    u[i] = u_alpha + u_gamma
                    v1 = agent_q
                    v_now = v1 + (np.array([u[i][0], u[i][1]]) * (1 / 48))
                    v_now = v_now / np.linalg.norm(v_now)

                    action[str(i)], _, _ = ctrl[i].computeControl(
                        control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                        cur_pos=obs[str(i)]["state"][0:3],
                        cur_quat=obs[str(i)]["state"][3:7],
                        cur_vel=obs[str(i)]["state"][10:13],
                        cur_ang_vel=obs[str(i)]["state"][13:16],
                        target_pos=obs[str(i)]["state"][0:3],
                        # same as the current position
                        target_rpy=np.array([0, 0, obs[str(i)]["state"][9]]),
                        # keep current yaw
                        target_vel=0.03 * env.MAX_SPEED_KMH * (
                                1000 / 3600) * np.abs(VP) * np.array([v_now[0], v_now[1], 0]))





            action[str(num_drones - 1)], _, _ = ctrl[num_drones - 1].computeControl(
                        control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                        cur_pos=obs[str(num_drones - 1)]["state"][0:3],
                        cur_quat=obs[str(num_drones - 1)]["state"][3:7],
                        cur_vel=obs[str(num_drones - 1)]["state"][10:13],
                        cur_ang_vel=obs[str(num_drones - 1)]["state"][13:16],
                        target_pos=obs[str(num_drones - 1)]["state"][0:3],
                        # same as the current position
                        target_rpy=np.array([0, 0, obs[str(num_drones - 1)]["state"][9]]),
                        # keep current yaw
                        target_vel=0.03 * env.MAX_SPEED_KMH * (
                                1000 / 3600) * np.abs(VE) * np.array([1, 0, 0]))






            for j in range(num_drones):
                logger.log(drone=j,
                           timestamp=x / env.SIM_FREQ,
                           state=obs[str(j)]["state"],
                           control=np.hstack(
                               [obs[str(j)]["state"][0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
                           # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                           )
            #### Printout ##############################################
        if x % env.SIM_FREQ == 0:
            env.render()
            #### Print matrices with the images captured by each drone #
            if vision:
                for j in range(num_drones):
                    print(obs[str(j)]["rgb"].shape, np.average(obs[str(j)]["rgb"]),
                          obs[str(j)]["dep"].shape, np.average(obs[str(j)]["dep"]),
                          obs[str(j)]["seg"].shape, np.average(obs[str(j)]["seg"])
                          )


            #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.TIMESTEP)

            #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid")  # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
        #### Define and parse (optional) arguments for the script ##
        parser = argparse.ArgumentParser(
            description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
        parser.add_argument('--drone', default=DEFAULT_DRONES, type=DroneModel, help='Drone model (default: CF2X)',
                            metavar='', choices=DroneModel)
        parser.add_argument('--num_drones', default=DEFAULT_NUM_DRONES, type=int, help='Number of drones (default: 3)',
                            metavar='')
        parser.add_argument('--physics', default=DEFAULT_PHYSICS, type=Physics, help='Physics updates (default: PYB)',
                            metavar='', choices=Physics)
        parser.add_argument('--vision', default=DEFAULT_VISION, type=str2bool,
                            help='Whether to use VisionAviary (default: False)', metavar='')
        parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool,
                            help='Whether to use PyBullet GUI (default: True)',
                            metavar='')
        parser.add_argument('--record_video', default=DEFAULT_RECORD_VISION, type=str2bool,
                            help='Whether to record a video (default: False)', metavar='')
        parser.add_argument('--plot', default=DEFAULT_PLOT, type=str2bool,
                            help='Whether to plot the simulation results (default: True)', metavar='')
        parser.add_argument('--user_debug_gui', default=DEFAULT_USER_DEBUG_GUI, type=str2bool,
                            help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
        parser.add_argument('--aggregate', default=DEFAULT_AGGREGATE, type=str2bool,
                            help='Whether to aggregate physics steps (default: True)', metavar='')
        parser.add_argument('--obstacles', default=DEFAULT_OBSTACLES, type=str2bool,
                            help='Whether to add obstacles to the environment (default: True)', metavar='')
        parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int,
                            help='Simulation frequency in Hz (default: 240)', metavar='')
        parser.add_argument('--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int,
                            help='Control frequency in Hz (default: 48)', metavar='')
        parser.add_argument('--duration_sec', default=DEFAULT_DURATION_SEC, type=int,
                            help='Duration of the simulation in seconds (default: 5)', metavar='')
        parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
                            help='Folder where to save logs (default: "results")', metavar='')
        parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool,
                            help='Whether example is being run by a notebook (default: "False")', metavar='')
        ARGS = parser.parse_args()

        run(**vars(ARGS))
























