"""
The module that contains all the necessary logic for the fermions.
"""
from jsonschema import validate
import numpy as np
from scipy.sparse.linalg import expm

NUM_WIRES = 8

exper_schema = {
    "type": "object",
    "required": ["instructions", "shots", "num_wires", "wire_order"],
    "properties": {
        "instructions": {"type": "array", "items": {"type": "array"}},
        "shots": {"type": "number", "minimum": 0, "maximum": 10 ** 3},
        "num_wires": {"type": "number", "minimum": 1, "maximum": 8},
        "seed": {"type": "number"},
        "wire_order": {"type": "string", "enum": ["interleaved"]},
    },
    "additionalProperties": False,
}

barrier_schema = {
    "type": "array",
    "minItems": 3,
    "maxItems": 3,
    "items": [
        {"type": "string", "enum": ["barrier"]},
        {
            "type": "array",
            "maxItems": NUM_WIRES,
            "items": [{"type": "number", "minimum": 0, "maximum": NUM_WIRES - 1}],
        },
        {"type": "array", "maxItems": 0},
    ],
}

load_measure_schema = {
    "type": "array",
    "minItems": 3,
    "maxItems": 3,
    "items": [
        {"type": "string", "enum": ["load", "measure"]},
        {
            "type": "array",
            "maxItems": 2,
            "items": [{"type": "number", "minimum": 0, "maximum": NUM_WIRES - 1}],
        },
        {"type": "array", "maxItems": 0},
    ],
}

hop_schema = {
    "type": "array",
    "minItems": 3,
    "maxItems": 3,
    "items": [
        {"type": "string", "enum": ["fhop"]},
        {
            "type": "array",
            "maxItems": 4,
            "items": [{"type": "number", "minimum": 0, "maximum": 7}],
        },
        {
            "type": "array",
            "items": [{"type": "number", "minimum": 0, "maximum": 6.284}],
        },
    ],
}

int_schema = {
    "type": "array",
    "minItems": 3,
    "maxItems": 3,
    "items": [
        {"type": "string", "enum": ["fint", "fphase"]},
        {
            "type": "array",
            "maxItems": 8,
            "items": [{"type": "number", "minimum": 0, "maximum": 7}],
        },
        {
            "type": "array",
            "items": [{"type": "number", "minimum": 0, "maximum": 6.284}],
        },
    ],
}


def check_with_schema(obj, schm):
    """
    Caller for the validate function.
    """
    # Fix this pylint issue whenever you have time, but be careful !
    # pylint: disable=W0703
    try:
        validate(instance=obj, schema=schm)
        return "", True
    except Exception as err:
        return str(err), False


def check_json_dict(json_dict):
    """
    Check if the json file has the appropiate syntax.

    Args:
        json_dict (dict): the dictonary that we will test.

    Returns:
        bool: is the expression having the appropiate syntax ?
    """
    ins_schema_dict = {
        "load": load_measure_schema,
        "barrier": barrier_schema,
        "fhop": hop_schema,
        "fint": int_schema,
        "fphase": int_schema,
        "measure": load_measure_schema,
    }
    max_exps = 50
    for expr in json_dict:
        err_code = "Wrong experiment name or too many experiments"
        # Fix this pylint issue whenever you have time, but be careful !
        # pylint: disable=W0702
        try:
            exp_ok = (
                expr.startswith("experiment_")
                and expr[11:].isdigit()
                and (int(expr[11:]) <= max_exps)
            )
        except:
            exp_ok = False
            break
        if not exp_ok:
            break
        err_code, exp_ok = check_with_schema(json_dict[expr], exper_schema)
        if not exp_ok:
            break
        ins_list = json_dict[expr]["instructions"]
        for ins in ins_list:
            # Fix this pylint issue whenever you have time, but be careful !
            # pylint: disable=W0703
            try:
                err_code, exp_ok = check_with_schema(ins, ins_schema_dict[ins[0]])
            except Exception as err:
                err_code = "Error in instruction " + str(err)
                exp_ok = False
            if not exp_ok:
                break
        if not exp_ok:
            break
    return err_code.replace("\n", ".."), exp_ok

def create_memory_data(shots_array, exp_name, n_shots):
    """
    The function to create memeory key in results dictionary
    with proprer formatting.
    """
    exp_sub_dict = {
        "header": {"name": "experiment_0", "extra metadata": "text"},
        "shots": 3,
        "success": True,
        "data": {"memory": None},  # slot 1 (Na)      # slot 2 (Li)
    }
    exp_sub_dict["header"]["name"] = exp_name
    exp_sub_dict["shots"] = n_shots
    memory_list = [
        str(shot).replace("[", "").replace("]", "").replace(",", "")
        for shot in shots_array
    ]
    exp_sub_dict["data"]["memory"] = memory_list
    return exp_sub_dict


def gen_circuit(json_dict):
    """The function the creates the instructions for the circuit.

    json_dict: The list of instructions for the specific run.
    """
    exp_name = next(iter(json_dict))
    ins_list = json_dict[next(iter(json_dict))]["instructions"]
    load_inds = []
    for inst in ins_list:
        if inst[0]=="load":
            load_inds.append(inst[1][0])
    n_shots = json_dict[next(iter(json_dict))]["shots"]
    if "seed" in json_dict[next(iter(json_dict))]:
        np.random.seed(json_dict[next(iter(json_dict))]["seed"])
    num_tweezers = int(NUM_WIRES/2)  # length of the tweezer array

    # calculate relevant parameters
    lattice_length = 2 * num_tweezers
    Nstates = 2 ** (lattice_length)
    psi = np.zeros(lattice_length)# important do not convert to complex yet!
    psi[load_inds] = 1.
    num_particles = len(load_inds)
    S_z = psi[0::2]-1*psi[1::2]
    S_z = 0.5*np.sum(S_z)# np.sum(S_z.real)

    # #### Build entire Fock Space basis
    # First in decimal representation

    dec_basis = np.arange(Nstates)
    # Now in binary representation.
    # See https://stackoverflow.com/a/21919812

    bin_basis = dec_basis[:,None] >> np.arange(2*num_tweezers)[::-1] & 1
    bin_basis = bin_basis.astype('int32')# Max 16 tweezers = 32 bits

    num_atoms_basis = np.sum(bin_basis, axis=1)
    req_num_hil_space = bin_basis[num_atoms_basis==num_particles]
    # #### Extract S_z sector
    # put minus sign on even indices for interleaved notation
    req_num_hil_space_signed = req_num_hil_space+0. #quick way for deep copy
    req_num_hil_space_signed[:,1::2] = -1.*req_num_hil_space[:,1::2]
    # calc S_z for each basis state in number Hilbert space
    S_z_hil_space = 0.5*np.sum(req_num_hil_space_signed, axis=1)

    # select basis states with required S_z
    req_sect_sub_space = req_num_hil_space[S_z_hil_space==S_z]
    req_sect_sub_space = req_sect_sub_space*1.0 #important, converting sector basis to float

    assert req_sect_sub_space.size>0, 'Empty sector' # Not really required but keep it. It can't be an empty sector as user choose a state there.

    # Invert the order of tweezers because we start counting from left but until now this code was starting from right.
    req_sect_sub_space = np.concatenate(np.split(req_sect_sub_space,num_tweezers,axis=1)[::-1],axis=1)
    # now modify psi to get it in the basis of this sector
    psi = 1.0*np.all(req_sect_sub_space==psi,axis=1)
    psi = psi+0j
    # ### Since now we have the basis for our sector, lets use it to build different gate generators

    # #### Build hopping matrix
    # Reasoning : hopping matrix element is 1 if two basis states differ at exactly two indices in their binary arrays.
    # Also these two indices must be from neighboring tweezers.
    def hopping_allowed(vec1, vec2, link=None):
        hop_allowed = False
        test = vec1-vec2
        diff_indices = np.where(test!=0.)[0]
        if diff_indices.size!=2:
            return hop_allowed
        if diff_indices[1]-diff_indices[0] != 2:
            return hop_allowed
        if link!=None:
            tweezers_inds = diff_indices//2
            if link != np.min(tweezers_inds):
                return hop_allowed
        hop_allowed = True
        return hop_allowed

    # Reasoning : hopping matrix element is 1 if two basis states differ at exactly two indices in their binary arrays.
    # Also these two indices must be from neighboring tweezers.
    def create_hop_array(link=None):
        J_arr = np.zeros((req_sect_sub_space.shape[0],req_sect_sub_space.shape[0]))
        up_tri_rows_inds, up_tri_col_inds = np.triu_indices(J_arr.shape[0], k = 1)

        for i in range(up_tri_rows_inds.size):
            row_ind = up_tri_rows_inds[i]
            col_ind = up_tri_col_inds[i]
            vec1 = req_sect_sub_space[row_ind]
            vec2 = req_sect_sub_space[col_ind]
            hop_allowed = hopping_allowed(vec1, vec2, link=link)
            J_arr[row_ind,col_ind] = hop_allowed

        J_arr = J_arr + J_arr.T
        return J_arr

    list_J_arrs = [create_hop_array(link) for link in range(num_tweezers-1)]

    # #### Build interaction matrix
    # Reasoning : interaction matrix element is on digonal and non zero for those states which have pair(s) of atoms at a site
    odd_cols = req_sect_sub_space[:,1::2]
    even_cols = req_sect_sub_space[:,0::2]
    num_pairs = np.sum(odd_cols*even_cols, axis=1)
    U_arr = np.diag(num_pairs)

    # #### Build phase matrices
    # Reasoning : phase matrix is digonal and non-zero for those states which have atleast one atom on the site where phase gate acts.
    list_phase_arrs =[]

    for i in range(lattice_length):
        site_occups = req_sect_sub_space[:,i]
        list_phase_arrs.append(np.diag(site_occups))

    ##############################################################
    ##############################################################
    measurement_indices = []
    shots_array = []
    for i in range(len(ins_list)):
        inst = ins_list[i]
        if inst[0] == "load":
            continue
        if inst[0] == "fhop":
            # the first two indices are the starting points
            # the other two indices are the end points
            hop_inds = inst[1]
            assert hop_inds[0]//2 == hop_inds[1]//2, 'First two wire inds are wrong'
            assert hop_inds[2]//2 == hop_inds[3]//2, 'Last two wire inds are wrong'
            start_tweezer = hop_inds[0]//2
            destination_tweezer = hop_inds[2]//2
            assert np.abs(destination_tweezer-start_tweezer) == 1, 'Only nearest neighbour hop allowed'
            link = start_tweezer
            if link>destination_tweezer:
                link = destination_tweezer
            theta = inst[2][0]
            Hhop = list_J_arrs[link]
            Uhop = expm(-1j * theta * Hhop)
            psi = np.dot(Uhop, psi)
        if inst[0] == "fint":
            # the first two indices are the starting points
            # the other two indices are the end points
            theta = inst[2][0]
            Uint = expm(-1j * theta * U_arr)
            psi = np.dot(Uint, psi)
        if inst[0] == "fphase":
            # the first two indices are the starting points
            # the other two indices are the end points
            Hphase = 0 * list_phase_arrs[0]
            for ii in inst[1]:  # np.arange(len(inst[1])):
                Hphase += list_phase_arrs[ii]
            theta = inst[2][0]
            Uphase = expm(-1j * theta * Hphase)
            psi = np.dot(Uphase, psi)
        if inst[0] == "measure":
            measurement_indices.append(inst[1][0])

    # first convert the result of reduced subspace to full space.
    # only give back the needed measurements
    if measurement_indices:
        probs = np.abs(psi) ** 2
        resultInd = np.random.choice(np.arange(req_sect_sub_space.shape[0]), p=probs, size=n_shots)

        measurements = np.zeros((n_shots, len(measurement_indices)), dtype=int)
        for j in range(n_shots):
            measurements[j] = req_sect_sub_space[resultInd[j]][measurement_indices]
        shots_array = measurements.tolist()

    exp_sub_dict = create_memory_data(shots_array, exp_name, n_shots)
    return exp_sub_dict


def add_job(json_dict, status_msg_dict):
    """
    The function that translates the json with the instructions into some circuit and executes it.
    It performs several checks for the job to see if it is properly working.
    If things are fine the job gets added the list of things that should be executed.

    json_dict: A dictonary of all the instructions.
    job_id: the ID of the job we are treating.
    """
    job_id = status_msg_dict["job_id"]

    result_dict = {
        "backend_name": "synqs_fermionic_tweezer_simulator",
        "backend_version": "0.0.1",
        "job_id": job_id,
        "qobj_id": None,
        "success": True,
        "status": "finished",
        "header": {},
        "results": [],
    }
    err_msg, json_is_fine = check_json_dict(json_dict)
    if json_is_fine:
        for exp in json_dict:
            exp_dict = {exp: json_dict[exp]}
            # Here we
            result_dict["results"].append(gen_circuit(exp_dict))

        status_msg_dict[
            "detail"
        ] += "; Passed json sanity check; Compilation done. Shots sent to solver."
        status_msg_dict["status"] = "DONE"
    else:
        status_msg_dict["detail"] += (
            "; Failed json sanity check. File will be deleted. Error message : "
            + err_msg
        )
        status_msg_dict["error_message"] += (
            "; Failed json sanity check. File will be deleted. Error message : "
            + err_msg
        )
        status_msg_dict["status"] = "ERROR"
    return result_dict, status_msg_dict
