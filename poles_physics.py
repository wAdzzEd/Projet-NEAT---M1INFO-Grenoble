import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc
from numba import njit
import warnings
from matplotlib.animation import FuncAnimation

def _normalize_display_angle(angle):
    """Normalise l'angle du pendule pour l'affichage."""
    return np.sign(angle)*(np.abs(angle) % (2.0 * np.pi)) 


_ACTION_COLOR = "tab:green"
MAX_FORCE=50



def _force_arrow_length(force, max_force=10.0, max_length=1.7):
    """Longueur (signée) de la flèche d'action, proportionnelle à la force."""
    return max_length * float(np.clip(force / max_force, -1.0, 1.0))


def _force_arrow_width(force, max_force=10.0, min_width=2.0, max_width=2.0):
    """Épaisseur de la flèche d'action, proportionnelle à la force."""
    t = float(np.clip(abs(force) / max_force, 0.0, 1.0))
    return min_width + (max_width - min_width) * t


@njit(cache=True)
def double_pole_dynamics_numba(state, applied_force, mc, m1, m2, l1, l2, gravity,friction_x, friction_theta1, friction_theta2):
    _, x_dot, theta1, theta1_dot, theta2, theta2_dot = state
    delta = theta1 - theta2
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s2, c2 = np.sin(theta2), np.cos(theta2)
    sd, cd = np.sin(delta), np.cos(delta)

    total_mass = mc + m1 + m2
    combined_mass = m1 + m2
    a = total_mass
    b = combined_mass * l1 * c1
    c = m2 * l2 * c2
    d = combined_mass * l1**2
    e = m2 * l1 * l2 * cd
    f = m2 * l2**2

    theta1_dot_sq = theta1_dot * theta1_dot
    theta2_dot_sq = theta2_dot * theta2_dot
    rhs0 = applied_force + combined_mass * l1 * theta1_dot_sq * s1 + m2 * l2 * theta2_dot_sq * s2 - friction_x * x_dot
    rhs1 = -m2 * l1 * l2 * theta2_dot_sq * sd - combined_mass * gravity * l1 * s1 - friction_theta1 * theta1_dot
    rhs2 = m2 * l1 * l2 * theta1_dot_sq * sd - m2 * gravity * l2 * s2 - friction_theta2 * theta2_dot

    d_reduced = d - b * b / a
    e_reduced = e - b * c / a
    f_reduced = f - c * c / a
    rhs1_reduced = rhs1 - b * rhs0 / a
    rhs2_reduced = rhs2 - c * rhs0 / a
    determinant = d_reduced * f_reduced - e_reduced * e_reduced

    if abs(determinant) < 1e-12:
        mass_matrix = np.array([[a, b, c], [b, d, e], [c, e, f]])
        accelerations = np.linalg.solve(mass_matrix, np.array([rhs0, rhs1, rhs2]))
        x_ddot, theta1_ddot, theta2_ddot = accelerations
    else:
        theta1_ddot = (rhs1_reduced * f_reduced - rhs2_reduced * e_reduced) / determinant
        theta2_ddot = (d_reduced * rhs2_reduced - e_reduced * rhs1_reduced) / determinant
        x_ddot = (rhs0 - b * theta1_ddot - c * theta2_ddot) / a

    return np.array((x_dot, x_ddot, theta1_dot, theta1_ddot, theta2_dot, theta2_ddot))


@njit(cache=True)
def double_pole_integrate_numba(state, applied_force, dt, mc, m1, m2, l1, l2, gravity, friction_x, friction_theta1, friction_theta2):
    k1 = double_pole_dynamics_numba(state, applied_force, mc, m1, m2, l1, l2, gravity, friction_x, friction_theta1, friction_theta2)
    k2 = double_pole_dynamics_numba(state + 0.5 * dt * k1, applied_force, mc, m1, m2, l1, l2, gravity, friction_x, friction_theta1, friction_theta2)
    k3 = double_pole_dynamics_numba(state + 0.5 * dt * k2, applied_force, mc, m1, m2, l1, l2, gravity, friction_x, friction_theta1, friction_theta2)
    k4 = double_pole_dynamics_numba(state + dt * k3, applied_force, mc, m1, m2, l1, l2, gravity, friction_x, friction_theta1, friction_theta2)
    state += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


@njit(cache=True)
def single_pole_dynamics_numba(state, applied_force, mc, m, length, gravity,friction_x, friction_theta):
    _, x_dot, theta, theta_dot = state
    sine = np.sin(theta)
    cosine = np.cos(theta)

    total_mass = mc + m
    coupling = m * length * cosine
    inertia = m * length**2
    rhs_cart = applied_force + m * length * theta_dot**2 * sine - friction_x * x_dot
    rhs_pole = -m * gravity * length * sine - friction_theta * theta_dot

    reduced_inertia = inertia - coupling * coupling / total_mass
    reduced_rhs = rhs_pole - coupling * rhs_cart / total_mass
    theta_ddot = reduced_rhs / reduced_inertia
    x_ddot = (rhs_cart - coupling * theta_ddot) / total_mass

    return np.array((x_dot, x_ddot, theta_dot, theta_ddot))


@njit(cache=True)
def single_pole_integrate_numba(state, applied_force, dt, mc, m, length, gravity,friction_x, friction_theta):
    k1 = single_pole_dynamics_numba(state, applied_force, mc, m, length, gravity,friction_x, friction_theta)
    k2 = single_pole_dynamics_numba(state + 0.5 * dt * k1, applied_force, mc, m, length, gravity,friction_x, friction_theta)
    k3 = single_pole_dynamics_numba(state + 0.5 * dt * k2, applied_force, mc, m, length, gravity,friction_x, friction_theta)
    k4 = single_pole_dynamics_numba(state + dt * k3, applied_force, mc, m, length, gravity,friction_x, friction_theta)
    state += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def warmup_numba():
    """Compile the single- and double-pole kernels before a simulation."""
    single_state = np.zeros(4, dtype=np.float64)
    double_state = np.zeros(6, dtype=np.float64)
    single_pole_integrate_numba(single_state, 0.0, 0.02, 2.0, 1.0, 1.0, 9.81, 0.1, 0.005)
    double_pole_integrate_numba(double_state, 0.0, 0.02, 2.0, 1.0, 1.0, 1.0, 1.0, 9.81, 0.1, 0.005, 0.005)
    
    
    
class SinglePoleEnv:
    """Chariot portant un pendule simple."""

    def __init__(self, initial_state=None, renderize=False, show_state=False,
                 show_action=False):
        self.renderize = renderize
        if renderize:
            warnings.warn("Le rendu ralentit considérablement la simulation.", UserWarning)
        self.dt = 0.02
        self.gravity = 9.81
        self.cart_mass = 2.0
        self.pole_mass = 1.0
        self.pole_length = 1.0
        self.max_position = 2.4
        self.max_force = MAX_FORCE
        self.friction_x = 0.1
        self.friction_theta = 0.005
        default_state = [0.0, 0.0, np.pi, 0.05]
        self.initial_state = self._validate_initial_state(
            default_state if initial_state is None else initial_state,
            expected_size=4,
        )
        self.state = self.initial_state.copy()
        self.step_count = 0
        self.show_state = show_state
        self.show_action = show_action
        self.last_action = 0.0
        
        #Rendering attributes
        self.figure = None
        self.axis = None
        self.cart_line = None
        self.pole_line = None
        self.joint_point = None
        self.end_point = None
        self.state_x_arrow = None
        self.state_x_text = None
        self.state_angle_arc = None
        self.state_angle_text = None
        self.state_angle_reference = None
        self.action_arrow = None
        self.action_text = None

    @staticmethod
    def _validate_initial_state(initial_state, expected_size):
        state = np.asarray(initial_state, dtype=np.float64)
        if state.shape != (expected_size,):
            raise ValueError(f"initial_state doit contenir {expected_size} valeurs")
        return state.copy()

    def reset(self):
        self.state = self.initial_state.copy()
        self.step_count = 0
        return self.state.astype(np.float32)

    def _integrate(self, applied_force):
        single_pole_integrate_numba(
            self.state, applied_force, self.dt,
            self.cart_mass, self.pole_mass,
            self.pole_length, self.gravity,
            self.friction_x, self.friction_theta
        )

    def step(self, action):
        applied_force = float(np.clip(action, -self.max_force, self.max_force))
        self.last_action = applied_force
        self._integrate(applied_force)
        self.step_count += 1
        x, _, _, _ = self.state
        terminated = bool(abs(x) > self.max_position)
        if self.renderize:
            self.render()
        return self.state.astype(np.float32),terminated

    def render(self):
        """Affiche le pendule, avec les variables d'état si demandé."""
        if self.figure is None:
            self.figure, self.axis = plt.subplots(figsize=(8, 5))
            self.axis.set_aspect('equal')
            plt.ion()
            self.axis.set(xlim=(-3, 3), ylim=(-1.1, 1.1), title="Pendule Simple")
            self.axis.axhline(0, color="black")
            self.cart_line, = self.axis.plot([], [], color="tab:blue", linewidth=8)
            self.pole_line, = self.axis.plot([], [], color="tab:red", linewidth=3)
            self.joint_point, = self.axis.plot([], [], "o", color="black")
            self.end_point, = self.axis.plot([], [], "o", color="black")
            
            self.time_text = self.axis.text(0.95, 0.95, '', transform=self.axis.transAxes, 
                                            horizontalalignment='right', verticalalignment='top',
                                            fontsize=12, fontweight='bold')
            self.state_x_arrow = self.axis.annotate(
                '', xy=(0, -0.28), xytext=(0, -0.28),
                arrowprops={'arrowstyle': '<->', 'color': 'tab:purple', 'lw': 1.8},
            )
            self.state_x_text = self.axis.text(0, -0.43, 'x', color='tab:purple',
                                               ha='center', va='center', fontsize=12,
                                               fontweight='bold')
            self.state_angle_arc = Arc((0, 0), 0.75, 0.75,
                                       theta1=0, theta2=0, color='tab:orange', lw=2)
            self.axis.add_patch(self.state_angle_arc)
            self.state_angle_reference, = self.axis.plot(
                [], [], color='tab:orange', linestyle='--', linewidth=1
            )
            self.state_angle_text = self.axis.text(0, 0, r'$\theta$', color='tab:orange',
                                                   ha='center', va='center', fontsize=12,
                                                   fontweight='bold')
            self.action_arrow = self.axis.annotate(
                '', xy=(0, 0), xytext=(0, 0),
                arrowprops={'arrowstyle': '-|>', 'color': 'tab:green', 'lw': 2},
            )
            self.action_text = self.axis.text(
                0.95, 0.05, '', transform=self.axis.transAxes,
                color=_ACTION_COLOR, ha='right', va='bottom', fontsize=12,
                fontweight='bold',
            )
        
        assert self.figure is not None
        assert self.cart_line is not None
        assert self.pole_line is not None
        assert self.joint_point is not None
        assert self.end_point is not None
        assert self.time_text is not None
        assert self.state_x_arrow is not None
        assert self.state_x_text is not None
        assert self.state_angle_arc is not None
        assert self.state_angle_reference is not None
        assert self.state_angle_text is not None
        assert self.action_arrow is not None
        assert self.action_text is not None
        x, _, theta, _ = self.state
        end_x = x + self.pole_length * np.sin(theta)
        end_y = -self.pole_length * np.cos(theta)
        self.cart_line.set_data([x - 0.3, x + 0.3], [0, 0])
        self.pole_line.set_data([x, end_x], [0, end_y])
        self.joint_point.set_data([x], [0])
        self.end_point.set_data([end_x], [end_y])
        
        time = self.step_count * self.dt
        self.time_text.set_text(f"t = {time:.2f} s")
        self.state_x_arrow.xy = (x, -0.28)
        self.state_x_text.set_position((x / 2, -0.43))
        self.state_angle_arc.center = (x, 0)
        self.state_angle_reference.set_data([x, x], [0, -0.5])
        display_theta = _normalize_display_angle(theta)
        vertical_angle = -90.0
        pole_angle = vertical_angle + np.degrees(display_theta)
        if display_theta >= 0.0:
            self.state_angle_arc.theta1 = vertical_angle
            self.state_angle_arc.theta2 = pole_angle
        else:
            self.state_angle_arc.theta1 = pole_angle
            self.state_angle_arc.theta2 = vertical_angle
        self.state_angle_text.set_text(rf'$\theta$')
        self.state_angle_text.set_position((x + 0.52 * np.sin(display_theta / 2),
                                            -0.52 * np.cos(display_theta / 2)))
        for artist in (self.state_x_arrow, self.state_x_text,
                       self.state_angle_arc, self.state_angle_reference,
                       self.state_angle_text):
            artist.set_visible(self.show_state)

        arrow_length = _force_arrow_length(self.last_action, self.max_force)
        self.action_arrow.xy = (x + arrow_length, 0)
        self.action_arrow.set_position((x, 0))
        assert self.action_arrow.arrow_patch is not None
        self.action_arrow.arrow_patch.set_color(_ACTION_COLOR)
        self.action_arrow.arrow_patch.set_linewidth(
            _force_arrow_width(self.last_action, self.max_force)
        )
        self.action_arrow.set_visible(self.show_action)
        self.action_text.set_text(f'Force : {self.last_action:.2f} N')
        self.action_text.set_visible(self.show_action)
        
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def close(self):
        if self.figure is not None:
            plt.close(self.figure)
            self.figure = None
            self.axis = None
            self.cart_line = None
            self.pole_line = None
            self.joint_point = None
            self.end_point = None
            self.state_x_arrow = None
            self.state_x_text = None
            self.state_angle_arc = None
            self.state_angle_text = None
            self.state_angle_reference = None
            self.action_arrow = None
            self.action_text = None



class DoublePoleEnv:
    """Chariot portant deux pendules en série."""

    def __init__(self, initial_state=None, renderize=False, show_state=False,
                 show_action=False):
        self.renderize = renderize
        if renderize:
            warnings.warn("Le rendu ralentit considérablement la simulation.", UserWarning)
        self.dt = 0.02
        self.gravity = 9.81
        self.cart_mass = 2.0
        self.m1 = 1.0
        self.m2 = 1.0
        self.l1 = 1.0
        self.l2 = 1.0
        self.max_position = 2.4
        self.max_force = MAX_FORCE
        self.friction_x = 0.1
        self.friction_theta1 = 0.005
        self.friction_theta2 = 0.005
        default_state = [0.0, 0.0, np.pi, 0.05, np.pi, -0.05]
        self.initial_state = self._validate_initial_state(
            default_state if initial_state is None else initial_state,
            expected_size=6,
        )
        self.state = self.initial_state.copy()
        self.step_count = 0
        self.last_action = 0.0
        
        #rendering attributes
        self.show_state = show_state
        self.show_action = show_action
        self.figure = None
        self.axis = None
        self.cart_line = None
        self.pole_1_line = None
        self.pole_2_line = None
        self.joint_point = None
        self.cart_joint_point = None
        self.end_point = None
        self.state_x_arrow = None
        self.state_x_text = None
        self.state_angle_arcs = []
        self.state_angle_texts = []
        self.state_angle_references = []
        self.action_arrow = None
        self.action_text = None

    @staticmethod
    def _validate_initial_state(initial_state, expected_size):
        state = np.asarray(initial_state, dtype=np.float64)
        if state.shape != (expected_size,):
            raise ValueError(f"initial_state doit contenir {expected_size} valeurs")
        return state.copy()

    def reset(self):
        self.state = self.initial_state.copy()
        self.step_count = 0
        return self.state.astype(np.float32)

    def _integrate(self, applied_force):
        double_pole_integrate_numba(
            self.state, applied_force, self.dt,
            self.cart_mass, self.m1, self.m2,
            self.l1, self.l2, self.gravity,
            self.friction_x, self.friction_theta1, 
            self.friction_theta2
        )


    def step(self, action):
        applied_force = float(np.clip(action, -self.max_force, self.max_force))
        self.last_action = applied_force
        self._integrate(applied_force)
        self.step_count += 1
        x, _, _, _, _, _ = self.state
        terminated = bool(abs(x) > self.max_position)
        if self.renderize:
            self.render()
        return self.state.astype(np.float32), terminated

    def render(self):
        """Affiche le pendule, avec les variables d'état si demandé."""
        if self.figure is None:
            self.figure, self.axis = plt.subplots(figsize=(8, 5))
            plt.ion()
            self.axis.set(xlim=(-3, 3), ylim=(-2.2, 2.2), title="Pendule Double")
            # AJOUT : Assure que le pendule ne se déforme pas visuellement
            self.axis.set_aspect('equal')
            
            self.axis.axhline(0, color="black")
            self.cart_line, = self.axis.plot([], [], color="tab:blue", linewidth=8)
            self.pole_1_line, = self.axis.plot([], [], color="tab:red", linewidth=3)
            self.pole_2_line, = self.axis.plot([], [], color="tab:green", linewidth=3)
            self.cart_joint_point, = self.axis.plot([], [], "o", color="black")
            self.joint_point, = self.axis.plot([], [], "o", color="black")
            self.end_point, = self.axis.plot([], [], "o", color="black")
            
            self.time_text = self.axis.text(0.95, 0.95, '', transform=self.axis.transAxes, 
                                            horizontalalignment='right', verticalalignment='top',
                                            fontsize=12, fontweight='bold')
            self.state_x_arrow = self.axis.annotate(
                '', xy=(0, -0.42), xytext=(0, -0.42),
                arrowprops={'arrowstyle': '<->', 'color': 'tab:purple', 'lw': 1.8},
            )
            self.state_x_text = self.axis.text(0, -0.57, 'x', color='tab:purple',
                                               ha='center', va='center', fontsize=12,
                                               fontweight='bold')
            for label in (r'$\theta_1$', r'$\theta_2$'):
                arc = Arc((0, 0), 0.75, 0.75, theta1=0, theta2=0,
                          color='tab:orange', lw=2)
                self.axis.add_patch(arc)
                self.state_angle_arcs.append(arc)
                reference, = self.axis.plot(
                    [], [], color='tab:orange', linestyle='--', linewidth=1
                )
                self.state_angle_references.append(reference)
                self.state_angle_texts.append(
                    self.axis.text(0, 0, label, color='tab:orange',
                                   ha='center', va='center', fontsize=12,
                                   fontweight='bold')
                )
            self.action_arrow = self.axis.annotate(
                '', xy=(0, 0), xytext=(0, 0),
                arrowprops={'arrowstyle': '-|>', 'color': 'tab:green', 'lw': 2},
            )
            self.action_text = self.axis.text(
                0.95, 0.05, '', transform=self.axis.transAxes,
                color=_ACTION_COLOR, ha='right', va='bottom', fontsize=12,
                fontweight='bold',
            )

        assert self.figure is not None
        assert self.cart_line is not None
        assert self.pole_1_line is not None
        assert self.pole_2_line is not None
        assert self.cart_joint_point is not None
        assert self.joint_point is not None
        assert self.end_point is not None
        assert self.time_text is not None
        assert self.state_x_arrow is not None
        assert self.state_x_text is not None
        assert self.action_arrow is not None
        assert self.action_text is not None
        cart_line = self.cart_line
        pole_1_line = self.pole_1_line
        pole_2_line = self.pole_2_line
        cart_joint_point = self.cart_joint_point
        joint_point = self.joint_point
        end_point = self.end_point
        time_text = self.time_text
        state_x_arrow = self.state_x_arrow
        state_x_text = self.state_x_text
        x, _, theta1, _, theta2, _ = self.state
        joint_x = x + self.l1 * np.sin(theta1)
        joint_y = -self.l1 * np.cos(theta1)
        end_x = joint_x + self.l2 * np.sin(theta2)
        end_y = joint_y - self.l2 * np.cos(theta2)
        
        cart_line.set_data([x - 0.3, x + 0.3], [0, 0])
        pole_1_line.set_data([x, joint_x], [0, joint_y])
        pole_2_line.set_data([joint_x, end_x], [joint_y, end_y])
        cart_joint_point.set_data([x], [0])
        joint_point.set_data([joint_x], [joint_y])
        end_point.set_data([end_x], [end_y])
        
        time = self.step_count * self.dt
        time_text.set_text(f"t = {time:.2f} s")
        state_x_arrow.xy = (x, -0.42)
        state_x_text.set_position((x / 2, -0.57))
        for i, (arc, reference, text, pivot_x, pivot_y, theta) in enumerate(zip(
            self.state_angle_arcs,
            self.state_angle_references,
            self.state_angle_texts,
            (x, joint_x),
            (0, joint_y),
            (theta1, theta2),
        )):
            arc.center = (pivot_x, pivot_y)
            reference.set_data([pivot_x, pivot_x], [pivot_y, pivot_y - 0.5])
            display_theta = _normalize_display_angle(theta)
            vertical_angle = -90.0
            pole_angle = vertical_angle + np.degrees(display_theta)
            if display_theta >= 0.0:
                arc.theta1 = vertical_angle
                arc.theta2 = pole_angle
            else:
                arc.theta1 = pole_angle
                arc.theta2 = vertical_angle
            text.set_text(rf'$\theta_{i + 1}$')
            text.set_position((pivot_x + 0.52 * np.sin(display_theta / 2),
                               pivot_y - 0.52 * np.cos(display_theta / 2)))
        for artist in (state_x_arrow, state_x_text,
                       *self.state_angle_arcs, *self.state_angle_references,
                       *self.state_angle_texts):
            artist.set_visible(self.show_state)

        arrow_length = _force_arrow_length(self.last_action, self.max_force)
        self.action_arrow.xy = (x + arrow_length, 0)
        self.action_arrow.set_position((x, 0))
        assert self.action_arrow.arrow_patch is not None
        self.action_arrow.arrow_patch.set_color(_ACTION_COLOR)
        self.action_arrow.arrow_patch.set_linewidth(
            _force_arrow_width(self.last_action, self.max_force)
        )
        self.action_arrow.set_visible(self.show_action)
        self.action_text.set_text(f'Force : {self.last_action:.2f} N')
        self.action_text.set_visible(self.show_action)
        
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()


    def close(self):
        if self.figure is not None:
            plt.close(self.figure)
            self.figure = None
            self.axis = None
            self.cart_line = None
            self.pole_1_line = None
            self.pole_2_line = None
            self.cart_joint_point = None
            self.joint_point = None
            self.end_point = None
            self.state_x_arrow = None
            self.state_x_text = None
            self.state_angle_arcs = []
            self.state_angle_texts = []
            self.state_angle_references = []
            self.action_arrow = None
            self.action_text = None


def animate_simulation(env, controller, duration=10.0, fps=50.0,
                       interval=None, show_state=False, show_action=False, speed=1.0):
    """Anime une simulation avec ``FuncAnimation`` et blitting (rendu efficace).

    Contrairement à ``env.render()``, qui redessine toute la figure à chaque
    pas, seuls les éléments graphiques modifiés sont rafraîchis
    (``blit=True``). C'est la méthode à privilégier pour produire des rendus.

    Parameters
    ----------
    env : SinglePoleEnv ou DoublePoleEnv
        Environnement à simuler. Son rendu interne est désactivé pendant
        l'animation puis restauré.
    controller : callable
        Fonction qui reçoit l'observation et renvoie la force à appliquer.
    duration : float
        Durée simulée, en secondes.
    fps : float
        Images par seconde de l'animation. Le temps simulé reste correct :
        chaque image avance de ``max(round(1 / (env.dt * fps)), 1)`` pas de
        simulation (le contrôleur est donc appelé à chaque pas, comme sans
        animation). Avec ``dt = 0.02`` s (50 pas/s), ``fps=50`` produit une
        image par pas ; ``fps=25`` saute une image sur deux et divise par
        deux le temps de génération.
    interval : float, optional
        Délai entre deux images, en millisecondes. Par défaut, la lecture se
        fait en temps réel, cadencée par ``1000 * pas_par_image * env.dt /
        speed``. Renseigner ``interval`` impose le cadencement directement.
    show_state : bool
        Affiche les variables d'état (position et angles) sur l'animation.
    show_action : bool
        Affiche la force appliquée sous forme de flèche (direction et
        longueur proportionnelles à la force et couleur verte constante).
    speed : float
        Facteur de vitesse de lecture (1.0 = temps réel). Ignoré si
        ``interval`` est renseigné.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        L'animation. Conservez une référence vers l'objet retourné tant
        qu'elle est affichée, sinon le ramasse-miettes peut la détruire.
    """

    previous_renderize = env.renderize
    env.renderize = False
    observation = env.reset()
    num_steps = max(int(round(duration / env.dt)), 1)
    steps_per_frame = max(int(round(1.0 / (float(fps) * env.dt))), 1)
    num_frames = (num_steps + steps_per_frame - 1) // steps_per_frame
    if interval is None:
        interval = 1000.0 * steps_per_frame * env.dt / max(float(speed), 1e-9)

    is_double = isinstance(env, DoublePoleEnv)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.set_aspect("equal")
    if is_double:
        axis.set(xlim=(-3, 3), ylim=(-2.2, 2.2), title="Pendule Double")
        arrow_y, text_y = -0.42, -0.57
        angle_labels = (r"$\theta_1$", r"$\theta_2$")
        pole_colors = ("tab:red", "tab:green")
    else:
        axis.set(xlim=(-3, 3), ylim=(-1.1, 1.1), title="Pendule Simple")
        arrow_y, text_y = -0.28, -0.43
        angle_labels = (r"$\theta$",)
        pole_colors = ("tab:red",)
    axis.axhline(0, color="black")  # fond statique : dessiné une seule fois

    def pole_points(state):
        """Points de la chaîne [chariot, ..., extrémité] et angles."""
        if is_double:
            x, _, theta1, _, theta2, _ = state
            joint = (x + env.l1 * np.sin(theta1), -env.l1 * np.cos(theta1))
            end = (joint[0] + env.l2 * np.sin(theta2),
                   joint[1] - env.l2 * np.cos(theta2))
            return [(x, 0.0), joint, end], (theta1, theta2)
        x, _, theta, _ = state
        end = (x + env.pole_length * np.sin(theta),
               -env.pole_length * np.cos(theta))
        return [(x, 0.0), end], (theta,)

    # Artistes dynamiques : seuls ceux-ci sont rafraîchis grâce au blitting.
    cart_line, = axis.plot([], [], color="tab:blue", linewidth=8, animated=True)
    pole_lines = [axis.plot([], [], color=color, linewidth=3, animated=True)[0]
                  for color in pole_colors]
    markers = [axis.plot([], [], "o", color="black", animated=True)[0]
               for _ in range(len(pole_colors) + 1)]
    time_text = axis.text(0.95, 0.95, "", transform=axis.transAxes,
                          horizontalalignment="right", verticalalignment="top",
                          fontsize=12, fontweight="bold", animated=True)
    state_x_arrow = axis.annotate(
        "", xy=(0, arrow_y), xytext=(0, arrow_y),
        arrowprops={"arrowstyle": "<->", "color": "tab:purple", "lw": 1.8},
        animated=True,
    )
    state_x_text = axis.text(0, text_y, "x", color="tab:purple", ha="center",
                             va="center", fontsize=12, fontweight="bold",
                             animated=True)
    state_angle_artists = []
    for label in angle_labels:
        arc = Arc((0, 0), 0.75, 0.75, theta1=0, theta2=0,
                  color="tab:orange", lw=2, animated=True)
        axis.add_patch(arc)
        reference, = axis.plot([], [], color="tab:orange", linestyle="--",
                               linewidth=1, animated=True)
        text = axis.text(0, 0, label, color="tab:orange", ha="center",
                         va="center", fontsize=12, fontweight="bold",
                         animated=True)
        state_angle_artists.append((arc, reference, text))
    state_artists = [state_x_arrow, state_x_text]
    for arc, reference, text in state_angle_artists:
        state_artists.extend((arc, reference, text))
    action_arrow = axis.annotate(
        "", xy=(0, 0), xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "color": "tab:green", "lw": 2},
        animated=True,
    )
    action_text = axis.text(
        0.95, 0.05, "", transform=axis.transAxes, color=_ACTION_COLOR,
        ha="right", va="bottom", fontsize=12, fontweight="bold", animated=True,
    )

    def draw(state):
        """Met à jour les artistes depuis l'état et les renvoie (blitting)."""
        points, angles = pole_points(state)
        x = points[0][0]
        cart_line.set_data([x - 0.3, x + 0.3], [0, 0])
        for line, (xa, ya), (xb, yb) in zip(pole_lines, points, points[1:]):
            line.set_data([xa, xb], [ya, yb])
        for marker, (px, py) in zip(markers, points):
            marker.set_data([px], [py])
        time_text.set_text(f"t = {env.step_count * env.dt:.2f} s")
        state_x_arrow.xy = (x, arrow_y)
        state_x_text.set_position((x / 2, text_y))
        for (arc, reference, text), (pivot_x, pivot_y), theta in zip(
                state_angle_artists, points, angles):
            arc.center = (pivot_x, pivot_y)
            reference.set_data([pivot_x, pivot_x], [pivot_y, pivot_y - 0.5])
            display_theta = _normalize_display_angle(theta)
            vertical_angle = -90.0
            pole_angle = vertical_angle + np.degrees(display_theta)
            if display_theta >= 0.0:
                arc.theta1, arc.theta2 = vertical_angle, pole_angle
            else:
                arc.theta1, arc.theta2 = pole_angle, vertical_angle
            text.set_position((pivot_x + 0.52 * np.sin(display_theta / 2),
                               pivot_y - 0.52 * np.cos(display_theta / 2)))
        arrow_length = _force_arrow_length(env.last_action, env.max_force)
        action_arrow.xy = (x + arrow_length, 0)
        action_arrow.set_position((x, 0))
        assert action_arrow.arrow_patch is not None
        action_arrow.arrow_patch.set_color(_ACTION_COLOR)
        action_arrow.arrow_patch.set_linewidth(
            _force_arrow_width(env.last_action, env.max_force)
        )
        action_text.set_text(f"Force : {env.last_action:.2f} N")
        return [cart_line, *pole_lines, *markers, time_text, *state_artists,
            action_arrow, action_text]

    terminated = False

    def init():
        for artist in state_artists:
            artist.set_visible(show_state)
        action_arrow.set_visible(show_action)
        action_text.set_visible(show_action)
        return draw(env.state)

    def update(_frame):
        nonlocal observation, terminated
        for _ in range(steps_per_frame):
            if terminated:
                break
            observation, terminated = env.step(controller(observation))
        return draw(env.state)

    animation = FuncAnimation(figure, update, frames=num_frames, init_func=init,
                              interval=interval, blit=True, repeat=False)
    env.renderize = previous_renderize
    animation.figure = figure  # type: ignore[attr-defined]  # exposé pour permettre plt.close(anim.figure) après usage
    return animation




__all__ = [
    "DoublePoleEnv",
    "SinglePoleEnv",
    "animate_simulation",
    "double_pole_dynamics_numba",
    "double_pole_integrate_numba",
    "single_pole_dynamics_numba",
    "single_pole_integrate_numba",
    "warmup_numba",
]
