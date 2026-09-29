import numpy as np

class MDPOscillationSolver:
    """
    Decision Engine: Solves Bellman Optimality Equation to compute optimal 
    precision docking actions while penalizing positional error and oscillation overshoot.
    """
    def __init__(self, gamma=0.95, alpha=0.5, beta=1.2):
        self.gamma = gamma          # Discount factor
        self.alpha = alpha          # Weight for displacement error penalty
        self.beta = beta            # Weight for oscillation (velocity) penalty
        
        # Action Space A: Discrete micro-correction thrust/steering vectors
        # Actions: [0: HOLD, 1: LEFT, 2: RIGHT, 3: UP, 4: DOWN]
        self.action_space = [
            np.array([0.0, 0.0]),    # HOLD
            np.array([-5.0, 0.0]),   # LEFT
            np.array([5.0, 0.0]),    # RIGHT
            np.array([0.0, -5.0]),   # UP
            np.array([0.0, 5.0])     # DOWN
        ]
        self.action_labels = ["HOLD/STABILIZE", "CORRECT LEFT", "CORRECT RIGHT", "CORRECT UP", "CORRECT DOWN"]

    def compute_reward(self, state, action_vector):
        """
        Evaluates Reward Function R(s, a) = - (alpha * ||error||^2 + beta * ||oscillation||^2)
        State S_t = [e_x, e_y, e_x_dot, e_y_dot]
        """
        ex, ey, ex_dot, ey_dot = state
        
        error_norm = np.sqrt(ex**2 + ey**2)
        oscillation_norm = np.sqrt(ex_dot**2 + ey_dot**2)
        
        # Base penalty for position error and dynamic oscillation rate
        penalty = (self.alpha * error_norm) + (self.beta * oscillation_norm)
        
        # Reward bonus if centered with zero velocity overshoot
        if error_norm < 10.0 and oscillation_norm < 2.0:
            reward = 100.0 - penalty
        else:
            reward = -penalty
            
        return float(reward)

    def solve_bellman_action(self, state):
        """
        Bellman Optimality Solver:
        Q*(s, a) = R(s, a) + gamma * V*(s')
        Computes action index that maximizes expected value.
        """
        best_action_idx = 0
        max_q_value = -float('inf')

        for idx, action_vec in enumerate(self.action_space):
            # Predict next state S' using kinematic transition model
            predicted_ex = state[0] + action_vec[0] + state[2]
            predicted_ey = state[1] + action_vec[1] + state[3]
            predicted_state = np.array([predicted_ex, predicted_ey, state[2], state[3]])

            # Evaluate immediate reward R(s, a)
            immediate_reward = self.compute_reward(state, action_vec)

            # Heuristic Future Value V*(s') estimation based on predicted offset
            future_value = -np.sqrt(predicted_ex**2 + predicted_ey**2)
            
            # Bellman Optimality Equation Value Calculation
            q_value = immediate_reward + (self.gamma * future_value)

            if q_value > max_q_value:
                max_q_value = q_value
                best_action_idx = idx

        optimal_action = self.action_space[best_action_idx]
        action_label = self.action_labels[best_action_idx]
        reward = self.compute_reward(state, optimal_action)

        return optimal_action, action_label, reward, max_q_value