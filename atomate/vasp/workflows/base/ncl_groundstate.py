
import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

class Agent:
    def __init__(self, pos_init, vel_init, nspacedim=3):
        self.ndim = len(pos_init)
        self.nspacedim = nspacedim

        # Position, Velocity, & Force
        self.pos = pos_init
        self.vel = vel_init
        self.force = 0.0*vel_init

        # Fitness (current) and Best fitness & position
        self.fit = -1.0
        self.fit_best = -1.0
        self.pos_best = pos_init

        # Params
        self.a = 0.5 # inertial
        self.c = 1.0 # cognition
        self.s = 2.0 # social

    def update_fitness(self, pes, fit_best_g, pos_best_g):
        self.fit = pes.evaluate_energy(self.pos)
        
        # Update personal best
        if (self.fit < self.fit_best) or self.fit_best == -1.0:
            self.fit_best = self.fit
            self.pos_best = self.pos.copy()
        
        # Update global best
        if (self.fit_best < fit_best_g[0]) or fit_best_g[0] == -1.0:
            fit_best_g[0] = self.fit_best
            for d in range(self.ndim):
                pos_best_g[d] = self.pos_best[d]

    def compute_force(self, pos_best_g, mass=1.0):
        # Inertial term
        self.force = (self.a - 1.0) * self.vel

        # Cognition term
        sigma_c = np.array([random.uniform(0.0, 1.0) for i in range(self.ndim)])
        self.force += self.c * sigma_c * (self.pos_best - self.pos)

        #Sigma_c = np.diag(sigma_c)
        #self.force += self.c * np.matmul(Sigma_c, self.pos_best - self.pos)

        # Social term
        sigma_s = np.array([random.uniform(0.0, 1.0) for i in range(self.ndim)])
        self.force += self.s * sigma_s * (pos_best_g - self.pos)

        #Sigma_s = np.diag(sigma_s)
        #self.force += self.s * np.matmul(Sigma_s, pos_best_g - self.pos)

        # Weight accordingly
        self.force *= mass

    def compute_force_spin(self, pos_best_g, mass=1.0):
        # Inertial term
        self.force = self.a * self.force
#         self.force = 0.0 * self.force

        # Cognition term
        sigma_c = random.uniform(0.0, 1.0)
        self.force += self.c * sigma_c * self.pos_best

        # Social term
        sigma_s = random.uniform(0.0, 1.0)
        self.force += self.s * sigma_s * pos_best_g

        # Weight accordingly
        self.force *= mass
        self.force /= npla.norm(self.force)

    def compute_forcebest_spin(self, pos_best_g, mass=1.0):

        nd = self.nspacedim
        for i in range(0, self.ndim, nd):
            F = np.array([random.gauss(0.0, 1.0) for i in range(nd)])
            F = (npla.norm(self.pos_best) / npla.norm(F)) * F
            self.force[i:i+nd] = F

        # Weight accordingly
        self.force *= mass
        self.force /= npla.norm(self.force)

    def compute_vel_stdPSO(self, pos_best_g, dt=1.0, mass=1.0):
        self.compute_force(pos_best_g, mass)
        self.vel = self.vel + dt * self.force

    def compute_vel_spinPSO(self, pos_best_g, gamma=1.0, lam=1.0, 
                            dt=1.0, mass=1.0):

        self.compute_force_spin(pos_best_g, mass)

        nd = self.nspacedim
        for i in range(0, self.ndim, nd):
            S = np.array(self.pos[i:i+nd])
            F = np.array(self.force[i:i+nd])
            self.vel[i:i+nd] = - gamma * np.cross(S, F)

        for i in range(0, self.ndim, nd):
            S = np.array(self.pos[i:i+nd])
            F = np.array(self.force[i:i+nd])
            self.vel[i:i+nd] += - lam * np.cross(S, np.cross(S, F))

        # Correct velocity
        pos_predict = self.pos + dt * self.vel
        pos_correct = pos_predict
        for i in range(0, self.ndim, nd):
            pos_correct[i:i+nd] *= npla.norm(self.pos[i:i+nd]) / npla.norm(pos_predict[i:i+nd])
        self.vel = (pos_correct - self.pos) / dt

    def compute_velbest_spinPSO(self, pos_best_g, grad, rho=1.0,
                                dt=1.0, mass=1.0):

        self.compute_forcebest_spin(pos_best_g, mass)
        nd = self.nspacedim
        for i in range(0, self.ndim, nd):
            S = np.array(self.pos[i:i+nd])
            F = np.array(self.force[i:i+nd])
            self.vel[i:i+nd] += - rho * np.cross(S, np.cross(S, F))

        self.vel = - rho*grad

        # print("Got here!")
        # print("rho = ", rho)

        # Correct velocity
        pos_predict = self.pos + dt * self.vel
        pos_correct = pos_predict
        nd = self.nspacedim
        for i in range(0, self.ndim, nd):
            pos_correct[i:i+nd] *= np.linalg.norm(self.pos[i:i+nd]) / np.linalg.norm(pos_predict[i:i+nd])
        self.vel = (pos_correct - self.pos) / dt

    def update_position(self, dt=1.0):
        self.pos = self.pos + dt * self.vel

    def get_position(self):
        return self.pos

    def get_fitness(self):
        return self.fit

class Swarm:
    def __init__(self, num_agent, positions_init, vels_init, nspacedim=3):
        self.num_agent = num_agent
        self.ndim = len(positions_init[0])
        self.nspacedim = nspacedim

        self.fit_best = [-1.0]
        self.pos_best = 0.0 * positions_init[0]

        self.index_best = -1
        self.rho = 0.01
        self.rho_lim = 5.0
        self.rho_scale = 0.8
        self.num_failure = 0
        self.num_success = 0
        self.failure_lim = 5
        self.success_lim = 5

        self.agents = []
        for i in range(num_agent):
            self.agents.append(Agent(positions_init[i], vels_init[i], self.nspacedim))

    def update_fitnesses(self, pes):
        for i in range(num_agent):
            self.agents[i].update_fitness(pes, self.fit_best, self.pos_best)
        #print('Best fitness = ', self.fit_best)
        #print('Best position = ', self.pos_best)

    def update_fitnesses_gcpso(self, pes):

        fit_best_old = self.fit_best.copy()

        for i in range(num_agent):
            self.agents[i].update_fitness(pes, self.fit_best, self.pos_best)

        #self.index_best = -1
        #self.fit_best[0] = -1.0
        for i in range(num_agent):
            if self.agents[i].get_fitness() <= self.fit_best[0] or self.fit_best[0] == -1.0:
                self.index_best = i
                self.fit_best[0] = self.agents[i].get_fitness()

        #tol = 1.0e-9
        #if np.abs(self.fit_best[0] - fit_best_old[0]) < tol:
        if self.fit_best[0] == fit_best_old[0]:
            self.num_failure += 1
            self.num_success = 0
        else:
            self.num_failure = 0
            self.num_success += 1

        if self.num_failure > self.failure_lim:
            self.rho *= self.rho_scale

        if self.num_success > self.success_lim and self.rho < self.rho_lim:
            self.rho *= 1.0/self.rho_scale
            #self.rho *= 1.0

#         print("fitness, old = ", self.fit_best[0], fit_best_old[0])
#         print("best fitness = ", self.fit_best[0])
#         print("best index = ", self.index_best)
#         print("rho = ", self.rho)
        print(self.fit_best[0])

    def compute_velocities(self, gamma=0.2, lam=0.5, dt=1.0, mass=1.0):
        for i in range(num_agent):
            #self.agents[i].compute_vel_stdPSO(self.pos_best, dt=dt, mass=mass)
            self.agents[i].compute_vel_spinPSO(self.pos_best, gamma=gamma, lam=lam, dt=dt, mass=mass)

    def compute_velocities_gcpso(self, pes, gamma=0.2, lam=0.5, dt=1.0, mass=1.0):
        for i in range(num_agent):
            if i == self.index_best:
                gradient = pes.compute_gradient_simple(self.pos_best)
                #self.agents[i].compute_velbest_stdPSO(self.pos_best, rho=self.rho, dt=dt, mass=mass)
                self.agents[i].compute_velbest_spinPSO(self.pos_best, gradient, rho=self.rho, dt=dt, mass=mass)
            else:
                #self.agents[i].compute_vel_stdPSO(self.pos_best, dt=dt, mass=mass)
                self.agents[i].compute_vel_spinPSO(self.pos_best, gamma=gamma, lam=lam, dt=dt, mass=mass)

    def update_positions(self, dt=1.0):
        for i in range(num_agent):
            self.agents[i].update_position(dt=dt)
    
    def get_positions(self):
        return [self.agents[i].get_position() for i in range(self.num_agent)]

    def get_pos_best(self):
        return self.pos_best


class OptimizerPSO:
    def __init__(self, pes, num_agent, 
                 positions_init, vels_init, 
                 nspacedim=3):
        self.pes = pes
        self.swarm = Swarm(num_agent, positions_init, vels_init, nspacedim)
    
    def optimize(self, pos_history, save_pos=False, savefreq=1, max_iter=100, dt=1.0, mass=1.0):
        for n in range(max_iter):
            self.swarm.update_fitnesses(self.pes)
            self.swarm.compute_velocities(gamma=0.2, lam=0.5, dt=dt, mass=mass)
            self.swarm.update_positions(dt)
            if save_pos and np.mod(n, savefreq) == 0:
                pos_history.append(self.swarm.get_positions())
    
    def optimize_gcpso(self, pos_history, grad_history, save_pos=False, savefreq=1, max_iter=100, dt=1.0, mass=1.0):
        for n in range(max_iter):
            self.swarm.update_fitnesses_gcpso(self.pes)
            self.swarm.compute_velocities_gcpso(self.pes, gamma=0.2, lam=0.5, dt=dt, mass=mass)
            self.swarm.update_positions(dt)
            if save_pos and np.mod(n, savefreq) == 0:
                pos_history.append(self.swarm.get_positions())
                grad_history.append(self.pes.compute_gradient_simple(self.swarm.get_pos_best()))


class NoncollinearConstrainSet(MPStaticSet):
    """
    """
    def __init__(self, structure, prev_incar=None, prev_kpoints=None,
                 reciprocal_density=100, small_gap_multiply=None, **kwargs):
        """
        Args:
            structure:
            prev_incar:
            prev_kpoints:
            reciprocal_density:
            small_gap_multiply:
            **kwargs:
        """

        super().__init__(structure, sort_structure=False, **kwargs)

        if isinstance(prev_kpoints, str):
            prev_kpoints = Kpoints.from_file(prev_kpoints)
        self.prev_kpoints = prev_kpoints

        self.reciprocal_density = reciprocal_density
        self.kwargs = kwargs
        self.small_gap_multiply = small_gap_multiply

    @property
    def incar(self):
        """
        """
        parent_incar = super().incar
        incar = Incar(parent_incar)
        
        incar.update({"ISYM": -1, "ISMEAR": 0, "LREAL":False, "LASPH":True})
        incar.update({"ISTART": 1})
        incar.pop("NSW", None)

        return incar

class NoncollinearConstrainFW(Firework):
    def __init__(self, structure=None, name="ncl_constrain",
                 vasp_input_set=None, vasp_input_set_params=None,
                 vasp_cmd=VASP_CMD, prev_calc_loc=True, prev_calc_dir=None,
                 db_file=DB_FILE, vasptodb_kwargs=None, parents=None,
                 additional_files=None,
                 **kwargs):
        """
        Standard static calculation Firework - either from a previous location or from a structure.

        Args:
            structure (Structure): Input structure. Note that for prev_calc_loc jobs, the structure 
                is only used to set the name of the FW and any structure with the same composition 
                can be used.
            name (str): Name for the Firework.
            vasp_input_set (VaspInputSet): input set to use (for jobs w/no parents)
                Defaults to MPStaticSet() if None.
            vasp_input_set_params (dict): Dict of vasp_input_set kwargs.
            vasp_cmd (str): Command to run vasp.
            prev_calc_loc (bool or str): If true (default), copies outputs from previous calc. If 
                a str value, retrieves a previous calculation output by name. If False/None, will create
                new static calculation using the provided structure.
            prev_calc_dir (str): Path to a previous calculation to copy from
            db_file (str): Path to file specifying db credentials.
            parents (Firework): Parents of this particular Firework. FW or list of FWS.
            vasptodb_kwargs (dict): kwargs to pass to VaspToDb
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """
        t = []

        vasp_input_set_params = vasp_input_set_params or {}
        vasptodb_kwargs = vasptodb_kwargs or {}
        if "additional_fields" not in vasptodb_kwargs:
            vasptodb_kwargs["additional_fields"] = {}
        vasptodb_kwargs["additional_fields"]["task_label"] = name

        fw_name = "{}-{}".format(structure.composition.reduced_formula if structure else "unknown", name)

        if prev_calc_dir:
            t.append(CopyVaspOutputs(calc_dir=prev_calc_dir, additional_files=additional_files,
                                     contcar_to_poscar=False))
        elif parents:
            if prev_calc_loc:
                t.append(CopyVaspOutputs(calc_loc=prev_calc_loc, additional_files=additional_files,
                                         contcar_to_poscar=False))

        if structure:
            vasp_input_set = vasp_input_set or NoncollinearConstrainSet(structure, **vasp_input_set_params)
            t.append(WriteVaspFromIOSet(structure=structure,
                                        vasp_input_set=vasp_input_set))
        else:
            raise ValueError("Must specify structure")

        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, auto_npar=">>auto_npar<<"))
        t.append(PassCalcLocs(name=name))
        t.append(
            VaspToDb(db_file=db_file, **vasptodb_kwargs))
        super(NoncollinearConstrainFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)

class SpinPSO_WF:
    def __init__(
        self,
        structure,
        static=False,
    ):
        """
        """

        self.uuid = str(uuid4())
        self.wf_meta = {
            "wf_uuid": self.uuid,
            "wf_name": self.__class__.__name__,
            "wf_version": __spin_pso_wf_version__,
        }
        self.static = static
