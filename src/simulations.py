import stim
import numpy as np

#from src.graph import get_3D_graph


class QECCodeSim:
    def __init__(self, repetitions, distance, p, n_shots, code_task, seed):
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

        self.compiled_sampler = self.circuit.compile_detector_sampler(seed=seed)
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

    def sample_syndromes(self, n_shots=None):
        if n_shots == None:
            n_shots = self.n_shots
        stim_data, observable_flips = self.compiled_sampler.sample(
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
        seed=None,
    ):
        super().__init__(repetitions, distance, p, n_shots, code_task, seed)
        self.code_task = code_task

    def syndrome_mask(self):
        sz = self.distance + 1

        syndrome_x = np.zeros((sz, sz), dtype=np.uint8)
        syndrome_x[::2, 1 : sz - 1 : 2] = 1
        syndrome_x[1::2, 2::2] = 1

        syndrome_z = np.rot90(syndrome_x) * 3

        return np.dstack([syndrome_x + syndrome_z] * (self.repetitions + 1))

    def generate_syndromes(self, use_for_mwpm=False, n_syndromes=None, n_shots=None):
        stabilizer_changes, flips, n_trivial_preds = super().sample_syndromes(n_shots)

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


