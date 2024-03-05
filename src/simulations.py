import stim
import numpy as np

from src.graph import get_3D_graph


class QECCodeSim:
    def __init__(self, repetitions, distance, p, n_shots, code_task):
        self.distance = distance
        self.repetitions = repetitions
        self.p = p
        self.n_shots = n_shots

        self.circuit = stim.Circuit.generated(
            code_task,
            rounds=repetitions,
            distance=distance,
            after_clifford_depolarization=p,
            after_reset_flip_probability=p,
            before_measure_flip_probability=p,
            before_round_data_depolarization=p,
        )

        # self.compiled_sampler = self.circuit.compile_detector_sampler(seed=seed)
        detector_coords, detector_indx = self.get_detector_coords()
        self.det_coords = detector_coords
        self.detector_indx = detector_indx
        
    def get_circuit(self):
        return self.circuit

    def get_detector_coords(self):
        # create detection grid for circuit
        det_coords = self.circuit.get_detector_coordinates()
        det_coords = np.array(list(det_coords.values()))

        # rescale space like coordinates:
        det_coords[:, :2] = det_coords[:, :2] / 2

        # convert to integers
        det_coords = det_coords.astype(np.uint8)

        # create a dictionary which gives detector X_i or Z_i given coordinate tuple(x, y, t)
        # False == Z and True == X in xz_map
        xz_map = (np.indices((self.distance + 1, self.distance + 1)).sum(axis=0) % 2).astype(bool)
        det_indx = np.arange(det_coords.shape[0])
        x_or_z = np.array([xz_map[cord[0], cord[1]] for cord in det_coords])
        
        x_dict = dict([(tuple(cord), ind) for cord, ind in zip(det_coords[x_or_z, :], det_indx[x_or_z])])
        z_dict = dict([(tuple(cord), ind) for cord, ind in zip(det_coords[~x_or_z, :], det_indx[~x_or_z])])
        
        detectors = {}
        detectors["x"] = x_dict
        detectors["z"] = z_dict
        
        return det_coords, detectors

    def sample_syndromes(self, seed=None, n_shots=None):
        
        sampler = self.circuit.compile_detector_sampler(seed=seed)
        if n_shots == None:
            n_shots = self.n_shots
        stim_data, observable_flips = sampler.sample(
            shots=n_shots,
            separate_observables=True,
        )

        # sums over the detectors to check if we have a parity change
        shots_w_flips = np.sum(stim_data, axis=1) != 0
        n_trivial_syndromes = np.invert(shots_w_flips).sum()

        # save only data for measurements with non-empty syndromes
        # but count how many trival (identity) syndromes we have
        stabilizer_changes = stim_data[shots_w_flips, :]
        flips = observable_flips[shots_w_flips, 0]

        return stabilizer_changes, flips.astype(np.uint8), n_trivial_syndromes


class RepetitionCodeSim(QECCodeSim):
    def __init__(
        self,
        repetitions,
        distance,
        p,
        n_shots,
        seed=None,
    ):
        super().__init__(
            repetitions, distance, p, n_shots, "repetition_code:memory", seed
        )
    
    def syndrome_mask(self):
        sz = self.distance 
        
class SurfaceCodeSim(QECCodeSim):
    def __init__(
        self,
        repetitions,
        distance,
        p,
        n_shots,
        code_task="surface_code:rotated_memory_z",
    ):
        super().__init__(repetitions, distance, p, n_shots, code_task)
        self.code_task = code_task

    def syndrome_mask(self):
        sz = self.distance + 1

        syndrome_x = np.zeros((sz, sz), dtype=np.uint8)
        syndrome_x[::2, 1 : sz - 1 : 2] = 1
        syndrome_x[1::2, 2::2] = 1

        syndrome_z = np.rot90(syndrome_x) * 3

        return np.dstack([syndrome_x + syndrome_z] * (self.repetitions + 1))

    def generate_syndromes(self, use_for_mwpm=False, seed=None, n_syndromes=None, n_shots=None):
        stabilizer_changes, flips, n_trivial_preds = super().sample_syndromes(seed, n_shots)

        mask = np.repeat(
            self.syndrome_mask()[None, ...], stabilizer_changes.shape[0], 0
        )
        syndromes = np.zeros_like(mask)
        syndromes[
            :, self.det_coords[:, 1], self.det_coords[:, 0], self.det_coords[:, 2]
        ] = stabilizer_changes
        
        syndromes[np.nonzero(syndromes)] = mask[np.nonzero(syndromes)]
        
        if use_for_mwpm:
            n_z = np.count_nonzero(syndromes == 3, axis=(1, 2, 3))
            n_x = np.count_nonzero(syndromes == 1, axis=(1, 2, 3))
            
            if self.code_task == "surface_code:rotated_memory_z":
                remove = (n_z > 0) < (n_x > 0)
            else:
                remove = (n_x > 0) < (n_z > 0)
            syndromes = syndromes[~remove, ...]
            flips = flips[~remove]
            n_trivial_preds += remove.sum()

        # make sure we get enough non-trivial syndromes if a certain number is desired
        if n_syndromes is not None:
            print("This might not work as expected right now!")
            while syndromes.shape[0] < n_syndromes:
                n_shots = n_syndromes - len(syndromes)
                new_syndromes, new_flips, new_n_trivial_preds = self.generate_syndromes(
                    n_shots=n_shots
                )
                syndromes = np.concatenate((syndromes, new_syndromes))
                flips = np.concatenate((flips, new_flips))
                n_trivial_preds += new_n_trivial_preds

            syndromes = syndromes[:n_syndromes]
            flips = flips[:n_syndromes]

        return syndromes, flips, n_trivial_preds

    def generate_batch(
        self,
        m_nearest_nodes,
        power,
        seed=None,
    ):
        """
        Generates a batch of graphs from a list of stim experiments.
        """
        batch = []
        stim_data_list = []
        observable_flips_list = []

        stim_data, observable_flips, n_trivial = self.sample_syndromes(seed, self.n_shots)
        mask = self.syndrome_mask()
        detector_coordinates = self.get_detector_coords()

        stim_data_list.extend(stim_data[: self.n_shots])
        observable_flips_list.extend(observable_flips[: self.n_shots])

        for i in range(len(stim_data_list)):
            # convert to syndrome grid:
            syndrome = self.stim_to_syndrome_3D(
                mask, detector_coordinates, stim_data_list[i]
            )
            # get the logical equivalence class:
            true_eq_class = np.array([int(observable_flips_list[i])])
            # map to graph representation
            graph = get_3D_graph(
                syndrome_3D=syndrome,
                target=true_eq_class,
                power=power,
                m_nearest_nodes=m_nearest_nodes,
                use_knn=False,
            )
            batch.append(graph)
        return batch, n_trivial

    def stim_to_syndrome_3D(self, mask, coordinates, stim_data):
        """
        Converts a stim detection event array to a syndrome grid.
        1 indicates a violated X-stabilizer, 3 a violated Z stabilizer.
        Only the difference between two subsequent cycles is stored.
        """
        # initialize grid:
        syndrome_3D = np.zeros_like(mask)

        # first to last time-step:
        syndrome_3D[coordinates[:, 1], coordinates[:, 0], coordinates[:, 2]] = stim_data

        # only store the difference in two subsequent syndromes:
        syndrome_3D[:, :, 1:] = (syndrome_3D[:, :, 1:] - syndrome_3D[:, :, 0:-1]) % 2

        # convert X (Z) stabilizers to 1(3) entries in the matrix
        syndrome_3D[np.nonzero(syndrome_3D)] = mask[np.nonzero(syndrome_3D)]

        return syndrome_3D
