import numpy as np
import casadi as ca

class MPCController:
    def __init__(self, N=5, dt=0.5, L=0.08, Q=np.diag([500, 500, 5000]), R=np.diag([0.1, 1000]), max_steer=1, max_vel=0.05):
        self.N = N  # Prediction horizon
        self.dt = dt
        self.L = L
        self.Q = Q  # State cost matrix
        self.R = R  # Input cost matrix
        self.max_steer = max_steer
        self.max_vel = max_vel

        # CasADi symbolic variables
        self.x = ca.SX.sym('x')
        self.y = ca.SX.sym('y')
        self.theta = ca.SX.sym('theta')
        self.v = ca.SX.sym('v')
        self.psi = ca.SX.sym('psi')

        # State and control vectors
        self.states = ca.vertcat(self.x, self.y, self.theta)  # X, Y, Theta
        self.controls = ca.vertcat(self.v, self.psi)  # Throttle, Steering

    def dynamics(self, X, U):
        """Define the system dynamics with non-linearities."""
        x = X[0]
        y = X[1]
        theta = X[2]
        v = U[0]  # Throttle
        psi = U[1]  # Steering

        dx = v * ca.cos(theta)
        #dx = -v * ca.sin(theta) - psi * v * ca.cos(theta) / self.L
        dy = v * ca.sin(theta)
        #dy = v * ca.cos(theta) - psi * v * ca.sin(theta) / self.L
        dtheta = v * ca.tan(psi) / self.L
        
        return ca.vertcat(dx, dy, dtheta)

    def generate_straight_trajectory(self, x_start, y_start, x_end, y_end, N=50):
        """Generate a straight line trajectory from start to end point."""
        ref = np.zeros((3, N))
        for k in range(N):
            t = k / (N - 1)
            ref[0, k] = x_start + t * (x_end - x_start)
            ref[1, k] = y_start + t * (y_end - y_start)
            ref[2, k] = np.arctan2(y_end - y_start, x_end - x_start)
        
        return ref
    
    def generate_curved_trajectory(self, x_start, y_start, theta_start, radius, direction, N=50):
        """Generate a curved trajectory with a given radius."""
        ref = np.zeros((3, N))
        ref[0, 0] = x_start
        ref[1, 0] = y_start
        ref[2, 0] = theta_start
        
        for k in range(1, N):
            angle_change = direction * k * np.pi / (N - 1)  # direction = 1 for left, -1 for right
            ref[2, k] = theta_start + angle_change  # Update angle along the curve
            ref[0, k] = x_start + radius * (np.sin(ref[2, k]) - np.sin(theta_start))
            ref[1, k] = y_start - radius * (np.cos(ref[2, k]) - np.cos(theta_start))
        
        return ref

    def solve_mpc(self, X0, ref):
        """Solve the MPC optimization problem."""
        opti = ca.Opti()  # Optimization problem
        
        # Decision variables for states and controls
        X = opti.variable(3, self.N+1)
        U = opti.variable(2, self.N)

        # Objective function
        cost = 0
        for k in range(self.N):
            state_error = X[:,k] - ref[:,k]
            cost += ca.mtimes([state_error.T, self.Q, state_error])
            control_input = U[:,k]
            cost += ca.mtimes([control_input.T, self.R, control_input])
            
            # Pénalité sur la dérivée de la commande de direction
            #if k > 0:
            #    delta_steering = U[1, k] - U[1, k-1]
            #    cost += 10000 * ca.mtimes(delta_steering.T, delta_steering)  # Ajustez le poids selon le besoin

        opti.minimize(cost)
        
        # Dynamics constraints
        for k in range(self.N):
            dynamics_output = self.dynamics(X[:,k], U[:,k])
            opti.subject_to(X[:,k+1] == X[:,k] + self.dt * dynamics_output)

        # Initial condition
        opti.subject_to(X[:,0] == X0)
        
        # Constraints on control inputs
        opti.subject_to(opti.bounded(-self.max_steer, U[1,:], self.max_steer))
        opti.subject_to(opti.bounded(0, U[0,:], self.max_vel))

        # Solve the optimization problem
        p_opts = {"expand": True}
        s_opts = {"max_iter": 1500}
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
        except Exception as e:
            print("Solver failed. Investigating values...")
            x_val = opti.debug.value(X)
            u_val = opti.debug.value(U)
            print(f"X values at failure: {x_val}")
            print(f"U values at failure: {u_val}")
            raise e
        
        return sol.value(U[:,0]), sol.value(X)
    
    def compute_control(self, X0, segment, radius=20):
        """Compute the optimal control input using MPC based on the circuit segment."""
        if segment == 'straight':
            # Generate a straight trajectory (example: between two curves)
            xd, yd = X0[0] + 5.0, X0[1]  # Adjust as needed for the specific circuit
            ref = self.generate_straight_trajectory(X0[0], X0[1], xd, yd, N=self.N+1)
        elif segment == 'curved_left':
            # Generate a curved trajectory with a given radius for left turns
            theta_start = X0[2]
            ref = self.generate_curved_trajectory(X0[0], X0[1], theta_start, radius, direction=1, N=self.N+1)
        elif segment == 'curved_right':
            # Generate a curved trajectory with a given radius for right turns
            theta_start = X0[2]
            ref = self.generate_curved_trajectory(X0[0], X0[1], theta_start, radius, direction=-1, N=self.N+1)
        else:
            raise ValueError("Unknown segment type. Must be 'straight', 'curved_left', or 'curved_right'.")

        control, _ = self.solve_mpc(X0, ref)
        return control[0], control[1]  # Throttle, Steering