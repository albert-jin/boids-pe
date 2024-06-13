import numpy as np
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary

class LeaderFollowerAviary(BaseMultiagentAviary):
	"""Multi-agent RL problem: leader-follower."""

	################################################################################

	def __init__(self,
				 drone_model: DroneModel=DroneModel.CF2X,
				 num_drones: int=2,
				 neighbourhood_radius: float=np.inf,
				 initial_xyzs=np.array([[.0,.0,0.4],[0,0,0.2]]),
				 initial_rpys=None,
				 physics: Physics=Physics.PYB,
				 freq: int=240,
				 aggregate_phy_steps: int=1,
				 gui=False,
				 record=False, 
				 obs: ObservationType=ObservationType.KIN,
				 act: ActionType=ActionType.RPM):
		"""Initialization of a multi-agent RL environment.

		Using the generic multi-agent RL superclass.

		Parameters
		----------
		drone_model : DroneModel, optional
			The desired drone type (detailed in an .urdf file in folder `assets`).
		num_drones : int, optional
			The desired number of drones in the aviary.
		neighbourhood_radius : float, optional
			Radius used to compute the drones' adjacency matrix, in meters.
		initial_xyzs: ndarray | None, optional
			(NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
		initial_rpys: ndarray | None, optional
			(NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
		physics : Physics, optional
			The desired implementation of PyBullet physics/custom dynamics.
		freq : int, optional
			The frequency (Hz) at which the physics engine steps.
		aggregate_phy_steps : int, optional
			The number of physics steps within one call to `BaseAviary.step()`.
		gui : bool, optional
			Whether to use PyBullet's GUI.
		record : bool, optional
			Whether to save a video of the simulation in folder `files/videos/`.
		obs : ObservationType, optional
			The type of observation space (kinematic information or vision)
		act : ActionType, optional
			The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

		"""
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

		################################################################################

		self.objectives=[(0,0,.7)]

		self.dynamic_objectives=self.objectives

		##################
		self.train_idx = 0   #构建一个全局变量用来标识训练的层次
		self.rewards_0 = float("NaN")  #记录0号机最终奖励大小
		self.rewards = list("NaN")  #记录追逐机最终奖励大小     
		self.rewards_sat = 0 #奖励满足次数，需要一个稳定的结果
		self.dimension = 3
	################################################################################
	def bound(self,i):
		states=np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

		for j in range(self.dimension):
			if states[i][j]>2 or states[i][j]<0.2:
				return True
		return False

	def _computeReward(self):
		"""Computes the current reward value(s).

        Returns
        -------
        dict[int, float]
            The reward value for each drone.

        """

		################################################################################

		rewards = {}

		################################################################################

		states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

		################################################################################
		if self.train_idx == 0:
			# 0号机初始训练
			if len(self.dynamic_objectives) <= 1:

				target_distance = np.linalg.norm(self.dynamic_objectives[0] - states[0, 0:3]) ** 2
				if (target_distance > 0.3):
					rewards[0] = -50000
				else:
					rewards[0] = -1 * target_distance

			if len(self.dynamic_objectives) > 1:

				target_distance = np.linalg.norm(self.dynamic_objectives[0] - states[0, 0:3]) ** 2

				if target_distance >= .1:
					rewards[0] = -1 * target_distance

				if target_distance < .1:
					rewards[0] = -1 * np.linalg.norm(self.dynamic_objectives[1] - states[0, 0:3]) ** 2 * 0.1 - 0.05
			if rewards[0] >= -0.5:  # 检查0号机训练是否满足条件
				self.rewards_sat += 1  # 若满足条件，满足次数加一
			else:
				self.rewards_sat = 0  # 一次不满足就从头开始计算

			if self.rewards_sat >= 1:  # 改参数视具体结果调整，意义为连续满足条件次数
				self.train_idx += 1
				self.rewards_sat = 0  # 清零，预备后面使用
				self.rewards_0 = rewards[0]  # 保存0号机最后一次奖励
			# for i in range(self.NUM_DRONES):
			# 	if self.bound(i):
			# 		# print("无人机out of bounds")
			# 		rewards[i] = -500000
			#
			# return rewards

		################################################################################

		elif self.train_idx == 1:
			# 剩下飞机的追逐训练

			distance = {}  # 设成list更简单

			rewards[0] = self.rewards_0

			for i in range(1, self.NUM_DRONES):

				distance[i] = np.linalg.norm(np.array([states[0, 0], states[0, 1], states[0, 2]]) - states[i, 0:3]) ** 2

				if (distance[i]) >= .1:  # 这个条件太苛刻了，调整一下

					rewards[i] = - 1 * np.linalg.norm(
						np.array([states[0, 0], states[0, 1], states[0, 2]]) - states[i, 0:3]) ** 2

				if (distance[i]) < .1:
					rewards[i] = - 1 * np.linalg.norm(
						np.array([states[0, 0], states[0, 1], states[0, 2]]) - states[i, 0:3]) ** 2 + 5000
			if min(distance.values()) < .1:  # 这个数字太小了，调整一下

				self.rewards_sat += 1
			else:
				self.rewards_sat = 0

			if self.rewards_sat >= 1:  # 改参数视具体结果调整，意义为连续满足条件次数
				self.train_idx += 1
				self.rewards_sat = 0  # 清零，预备后面使用
				self.rewards = rewards

			# for i in range(self.NUM_DRONES):
			# 	if self.bound(i):
			# 		# print("无人机out of bounds")
			# 		rewards[i] = -500000
			#
			# return rewards

		################################################################################

		elif self.train_idx == 2:
			# 0号机逃离训练
			distance = {}
			rewards_0 = 0
			for i in range(1, self.NUM_DRONES):

				distance[i] = np.linalg.norm(np.array([states[0, 0], states[0, 1], states[0, 2]]) - states[i, 0:3]) ** 2

				if (distance[i]) <= .1:  # 这个条件太苛刻了，调整一下

					rewards_0 += np.linalg.norm(
						np.array([states[0, 0], states[0, 1], states[0, 2]]) - states[i, 0:3]) ** 2

				if (distance[i]) > .1:
					rewards_0 += np.linalg.norm(
						np.array([states[0, 0], states[0, 1], states[0, 2]]) - states[i, 0:3]) ** 2 + 300
			rewards = self.rewards
			rewards[0] = rewards_0
			if rewards_0 >= 300:  # 根据实验调整
				self.rewards_sat += 1
			else:
				self.rewards_sat = 0
			if self.rewards_sat >= 1:  # 改参数视具体结果调整，意义为连续满足条件次数
				self.train_idx = 1  # 再次训练剩下飞机，也可以再设立一个标志位，不让其无限循环下去
				self.rewards_sat = 0  # 清零，预备后面使用
				self.rewards_0 = rewards[0]
				self.rewards = rewards
			# for i in range(self.NUM_DRONES):
			# 	if self.bound(i):
			# 		# print("无人机out of bounds")
			# 		rewards[i] = -500000
			# return rewards

	################################################################################
	
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

		distance={}

		for i in range(1, self.NUM_DRONES):

			distance[i] =  np.linalg.norm(np.array([states[0, 0], states[0, 1], states[0, 2]]) - states[i, 0:3])**2

		if min(distance.values())<.01:

			bool_val=True

		elif self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC:

			bool_val=True

		else:

			bool_val=False

		################################################################################

		done = {i: bool_val for i in range(self.NUM_DRONES)}

		done["__all__"] = bool_val # True if True in done.values() else False

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

		MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
		MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

		MAX_PITCH_ROLL = np.pi # Full range

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
		normalized_y = state[9] / np.pi # No reason to clip
		normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
		normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
		normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

		norm_and_clipped = np.hstack([normalized_pos_xy,
									  normalized_pos_z,
									  state[3:7],
									  normalized_rp,
									  normalized_y,
									  normalized_vel_xy,
									  normalized_vel_z,
									  normalized_ang_vel,
									  state[16:20]
									  ]).reshape(20,)

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
		if not(clipped_pos_xy == np.array(state[0:2])).all():
			print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
		if not(clipped_pos_z == np.array(state[2])).all():
			print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
		if not(clipped_rp == np.array(state[7:9])).all():
			print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
		if not(clipped_vel_xy == np.array(state[10:12])).all():
			print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
		if not(clipped_vel_z == np.array(state[12])).all():
			print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
