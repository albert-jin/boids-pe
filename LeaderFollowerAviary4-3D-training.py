from gym_pybullet_drones.utils.enums import DroneModel,Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
import numpy as np
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary
class LeaderFollowerAviary(BaseMultiagentAviary):
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=2,
                 neighbourhood_radius: float = np.inf,
                 # initial_xyzs = np.array([[0,-0.05,0.4],[0,0.05,0.3],[0,0.1,0.5],[0,-0.1,0.5],[0,0,0.5]]),
                 initial_xyzs=np.array([[0, -0.05, 0.4], [0, 0.05, 0.3]]),
                 initial_rpys = None,
                 physics: Physics = Physics.PYB,
                 freq: int = 240,
                 aggregate_phy_steps: int = 1,
                 gui = False,
                 record = False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM
    ):
        super().__init__(drone_model=drone_model,
                       num_drones=num_drones,
                       neighbourhood_radius=neighbourhood_radius,
                       initial_xyzs=initial_xyzs,
                       initial_rpys=initial_rpys,
                       physics=physics,
                       freq=freq,
                       aggregate_phy_steps=aggregate_phy_steps,
                       gui=gui,
                       record=record,
                       obs=obs,
                       act=act
                       )
        self.objectives=np.array([0,0,1])
        self.train_index=0
        self.max_bound=5
        self.min_bound=0.05
        self.num_escape_drones=1
        self.num_chase_drones=1
        self.which_to_train=0
        self.is_satisfy=False
        self.dimension=3
        # self.dis_before=initial_xyzs


    def is_outofbounds(self,i):
        states=np.array(self._getDroneStateVector(i))
        for j in range(self.dimension):
            if states[j]>self.max_bound or states[j]<self.min_bound:
                return True
        return False
    def escape_center(self):
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        center=np.zeros(3)
        for i in range(self.num_escape_drones):
            center+=states[i,0:3]
        return center
    def chase_center(self):
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        center = np.zeros(3)
        for i in range(self.num_escape_drones,self.NUM_DRONES):
            center += states[i, 0:3]
        return center
    def disfestochase(self):
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        d=np.linalg.norm(self.chase_center()/self.num_chase_drones-self.escape_center()/self.num_chase_drones)
        d1=np.linalg.norm(self.before[0,0:3]-self.before[1,0:3])
        return d-d1




    def _computeReward(self,test=False):
        if test==True:
            rewards = {}
            escape_center = self.escape_center() / self.num_escape_drones
            chase_center = self.chase_center() / self.num_chase_drones
            states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
            if test == True:
                for i in range(self.num_escape_drones):
                    rewards[i] = np.linalg.norm(states[i, 0:3] - chase_center)
                for j in range(self.num_escape_drones, self.NUM_DRONES):
                    rewards[j] = np.linalg.norm(states[j, 0:3] - escape_center)
            for t in range(self.NUM_DRONES):
                if self.is_outofbounds(t):
                    rewards[t] = -500
            self.before=np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
            return rewards

        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        rewards={}
        if self.train_index==0:
             # center_escape=self.escape_center()/self.num_escape_drones
             for i in range(self.num_escape_drones):
                 if (np.linalg.norm(states[i,0:3]-self.objectives))>0.2:
                     rewards[i]=-10000
                 else:
                     rewards[i]=-np.linalg.norm(states[i,0:3]-self.objectives)
             # for j in range(self.num_escape_drones,self.NUM_DRONES):
             #     rewards[j]=0
             #     if (np.linalg.norm(states[j,0:3]-center_escape)>0.2):
             #         rewards[j]=-1000
             #     else:
             #         rewards[j]=-np.linalg.norm(states[j,0:3]-center_escape)
             for t in range(self.NUM_DRONES):
                 if self.is_outofbounds(t):
                     rewards[t]=-10000
             for z in range(self.num_escape_drones):
                 if rewards[z]<-0.1:
                     self.train_index=0
                     self.which_to_train=0
                     self.before = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
                     return rewards
             self.train_index=1
             self.which_to_train=1
             self.before=np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
             return rewards

        elif self.train_index==1:
            center_escape=self.escape_center()/self.num_escape_drones
            center_chase=self.chase_center()/self.num_chase_drones
            # for i in range(self.num_escape_drones):
            #     rewards[i]=np.linalg.norm(states[i,0:3]-center_chase)
            for j in range(self.num_escape_drones,self.NUM_DRONES):
                if np.linalg.norm(states[j,0:3]-center_escape)<0.05:
                    rewards[j]=1000
                    # rewards[i]=-np.linalg.norm(states[i,0:3]-center_escape)*(-3000)
                else:
                    rewards[j]=10*self.disfestochase()
            for t in range(self.NUM_DRONES):
                if self.is_outofbounds(t):
                    rewards[t] = -50000
            for z in range(self.num_escape_drones,self.NUM_DRONES):
                if np.linalg.norm(states[z,0:3]-center_escape) <-0.1:
                    self.train_index = 1
                    self.which_to_train = 0
                    self.before = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
                    return rewards
            self.train_index = 2
            self.which_to_train = 1
            self.before = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
            return rewards
        elif self.train_index==2:
            center_escape = self.escape_center() / self.num_escape_drones
            center_chase = self.chase_center() / self.num_chase_drones
            for i in range(self.num_escape_drones):
                if (np.linalg.norm(states[i,0:3]-center_chase)>1.5):
                    rewards[i]=1000
                    # rewards[i]=np.linalg.norm(states[i,0:3]-center_chase)*100
                else:
                    rewards[i] = -self.disfestochase()

            # for j in range(self.num_escape_drones, self.NUM_DRONES):
            #     rewards[i] = -np.linalg.norm(states[j, 0:3] - center_escape)
            for t in range(self.NUM_DRONES):
                if self.is_outofbounds(t):
                    rewards[t] = -50000
            for z in range(self.num_escape_drones):
                if (np.linalg.norm(states[i,0:3]-center_chase))<0.5:
                    self.train_index = 2
                    self.which_to_train = 0
                    self.before = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
                    return rewards
            self.train_index = 1
            self.which_to_train = 1
            self.before = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
            return rewards

    def _computeDone(self):

        """Computes the current done value(s).

        Returns
        -------
        dict[int | "__all__", bool]
            Dictionary with the done value of each drone and
            one additional boolean value for key "__all__".

        """

        ################################################################################

        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

        distance = {}

        for i in range(1, self.NUM_DRONES):
            distance[i] = np.linalg.norm(np.array([states[0, 0], states[0, 1], states[0, 2]]) - states[i, 0:3]) ** 2

        if min(distance.values()) < .01:

            bool_val = True

        elif self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC:

            bool_val = True

        else:

            bool_val = False

        ################################################################################

        done = {i: bool_val for i in range(self.NUM_DRONES)}

        done["__all__"] = bool_val  # True if True in done.values() else False

        return done

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[int, dict[]]
            Dictionary of empty dictionaries.

        """
        return {i: {} for i in range(self.NUM_DRONES)}

    ################################################################################

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi  # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16] / np.linalg.norm(state[13:16]) if np.linalg.norm(
            state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20, )

        return norm_and_clipped

    ################################################################################

    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.

        """
        if not (clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(
                      state[0], state[1]))
        if not (clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not (clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(
                      state[7], state[8]))
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(
                      state[10], state[11]))
        if not (clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter,
                  "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))















            







