import os
import shutil
import itertools
import re
import shlex, subprocess
from tqdm import tqdm
from string import Template, ascii_uppercase

# Base command to execute for all simulations
IRSIM_BASE_CMD = "irsim -E $expno -p $paramfile -v"
# Base directory where special ParamFiles are found
PARAM_AUT_BASE_DIR = "./autParamFiles"
TRAINING_DIR_PARENT = "../training"
TMP_PARAMFILE = "tmpParam"
# Amount of time we want the simulation to happen
IRSIM_BASE_EXEC_TICKS = 2000
# Types of fitness functions, used to navigate the directories
fitness_funcs = ["stdev", "schmitt"]
# NNs archs available to test
nn_archs = ["irsim-ANN1", "irsim-ANN2", "irsim-ANN3", "irsim-CTRNNosc", "irsim-CTRNNosc-obs"] 

config_params = {
    "d": {
        "runtime": IRSIM_BASE_EXEC_TICKS,
        "xpos1": 0.6,
        "ypos1": 0.6,
        "xpos2": -0.6,
        "ypos2": -0.6
    },
    "v": {
        "runtime": IRSIM_BASE_EXEC_TICKS,
        "xpos1": 0.6,
        "ypos1": 0.6,
        "xpos2": 0.6,
        "ypos2": -0.6
    },
    "h": {
        "runtime": IRSIM_BASE_EXEC_TICKS,
        "xpos1": 0.6,
        "ypos1": 0.6,
        "xpos2": -0.6,
        "ypos2": 0.6
    }
}

file_mappings = {
    ".*ANN1": ["iriNeuralANN1.txt", 21],
    ".*ANN2": ["iriNeuralANN2.txt", 21],
    ".*ANN3": ["iriNeuralANN3.txt", 21],
    ".*CTRNN.*": ["iriNeuralCTRNNosc.txt", 24]
}

try:
    if not os.path.isdir(PARAM_AUT_BASE_DIR):
        raise FileNotFoundError
except FileNotFoundError:
    print(f"The paramfile directory for automations ${PARAM_AUT_BASE_DIR} was not found")
    exit(1)

try:
    if not os.path.isdir(TRAINING_DIR_PARENT):
        raise FileNotFoundError
except FileNotFoundError:
    print(f"The training experiments directory was not found")
    exit(1)

def purge_dir_files(directory: str) -> None:
    try:
        for root, dirs, files in os.walk(directory):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
    except FileNotFoundError as err:
        print(f"Directory already not present, skipping...")

def pair_match(redict: dict, item: str) -> list:
    for k in redict:
        if re.match(k, item):
            return redict[k]

# Tuples formed from the combinations of the different experiments
exps_comb_tuples = list(itertools.product(fitness_funcs, nn_archs))
exps_comb_proc = []

for i, tup in enumerate(exps_comb_tuples):
    if tup[0] == "stdev" and ((not tup[1].startswith("irsim-ANN")) or (tup[1] == "irsim-ANN3")):
        pass
    else:
        exps_comb_proc.append(tup)
del exps_comb_tuples

pairiter = tqdm(exps_comb_proc)
for pair in pairiter:
    cwd = os.path.join(TRAINING_DIR_PARENT, pair[0], pair[1])
    pairiter.set_description(f"sim {pair[0]}/{pair[1]}")
    #print(f"Working now in {cwd}")
    outputf_dir = "./outputFiles"
    genetic_dir = "./geneticDataFiles"
    #print(f"Current outputFiles dir: {outputf_dir}")
    #copy over the genetic data files
    src_genetics = os.path.join(cwd, "geneticDataFiles")
    if os.path.isdir(genetic_dir):
        pairiter.write(f"Directory {genetic_dir} exists, deleting...")
        shutil.rmtree(genetic_dir)
    shutil.copytree(src_genetics, genetic_dir)

    skewiter = tqdm(["d", "v", "h"], leave=False)
    for skew in skewiter:
        skewiter.set_description(f"scenario {skew}")
        purge_dir_files(outputf_dir)
        selected_match = pair_match(file_mappings, pair[1])
        #print(f"###### Modifying paramfile {selected_match[0]}")
        with open(os.path.join(PARAM_AUT_BASE_DIR, selected_match[0]), "r") as fr:
            src = Template(fr.read())
            result = src.substitute(config_params[skew])
            with open(TMP_PARAMFILE, "w") as fw:
                fw.write(result)
        # we have a complete config ready to simulate at TMP_PARAMFILE
        sim_bin = Template(IRSIM_BASE_CMD).substitute(expno=selected_match[1], paramfile=TMP_PARAMFILE)
        binpath = os.path.join(cwd, sim_bin)
        cmd = shlex.split(binpath)

        #print(f"------> Running command {cmd}")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL)
        proc.wait() # blocking call
        #print(f"<------ Finished experiment {pair[0]}/{pair[1]}")
        result_dir = f"{outputf_dir}_{pair[0]}_{pair[1]}_{skew}"
        #print(f"COPY: {outputf_dir} to {result_dir}")
        if os.path.isdir(result_dir):
            skewiter.write(f"Directory {result_dir} exists, deleting...")
            shutil.rmtree(result_dir)
        shutil.copytree(outputf_dir, result_dir)
        
    chiter = tqdm(["best5", "best10", "best50", "currentbest"], leave=False)
    for ch in chiter:
        chiter.set_description(f"scenario {ch}")
        purge_dir_files(outputf_dir)
        selected_match = pair_match(file_mappings, pair[1])
        with open(os.path.join(PARAM_AUT_BASE_DIR, selected_match[0]), "r") as fr:
            src = Template(fr.read())
            result = src.substitute(config_params["d"])
            with open(TMP_PARAMFILE, "w") as fw:
                fw.write(result)
        # we have a complete config ready to simulate at TMP_PARAMFILE
        suffix = f" -c ./geneticDataFiles/{ch}" if ch == "currentbest" else f" -c ./geneticDataFiles/{ch}.log"
        sim_bin = Template(IRSIM_BASE_CMD).substitute(expno=selected_match[1], paramfile=TMP_PARAMFILE) + suffix
        binpath = os.path.join(cwd, sim_bin)
        cmd = shlex.split(binpath)

        #print(f"------> Running command {cmd}")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL)
        proc.wait() # blocking call
        #print(f"<------ Finished experiment {pair[0]}/{pair[1]}")
        result_dir = f"{outputf_dir}_{pair[0]}_{pair[1]}_{ch}"
        #print(f"COPY: {outputf_dir} to {result_dir}")
        if os.path.isdir(result_dir):
            chiter.write(f"Directory {result_dir} exists, deleting...")
            shutil.rmtree(result_dir)
        shutil.copytree(outputf_dir, result_dir)

        

