import pathlib
import argparse
import os.path
import random
import itertools
import getpass


def mkdir(path, mode=0o777):
    pathlib.Path(path).mkdir(mode=mode, parents=True, exist_ok=True)


if __name__ == "__main__":
    num_runs_per_job = 1
    mkdir("logs")
    launch_jobs_tmp = [
        ## CIFAR10 - Chau
        #    "python main.py -c configs/CIFAR10/template.json --traintools train_synthesized --noise_ratio 0.2 --noise_type asym",
        #    "python main.py -c configs/CIFAR10/template.json --traintools train_synthesized --noise_ratio 0.4 --noise_type asym",
        #    "python main.py -c configs/CIFAR10/template.json --traintools train_synthesized --noise_ratio 0.2 --noise_type sym",
        #    "python main.py -c configs/CIFAR10/template.json --traintools train_synthesized --noise_ratio 0.4 --noise_type sym",
        #  ## CIFAR100
        # "python main.py -c configs/cifar100.json --traintools train_synthesized --noise_ratio 0.2 --noise_type asym",
        # "python main.py -c configs/cifar100.json --traintools train_synthesized --noise_ratio 0.4 --noise_type asym",
        # "python main.py -c configs/cifar100.json --traintools train_synthesized --noise_ratio 0.2 --noise_type sym",
        # "python main.py -c configs/cifar100.json --traintools train_synthesized --noise_ratio 0.4 --noise_type sym",
        ## CP
        # "python main.py -c configs/CP.json --traintools train_realworld",
        ## Animal10N
        # "python main.py -c configs/animal10N.json --traintools train_realworld",
        ## clothing
        "python main.py -c configs/clothing.json --warmup 1 --traintools train_realworld"
    ] * num_runs_per_job

    ## grid search parameters CIFAR10
    # estimation_method = ["dualT", "total_variation", "BLTM", "growing_cluster", "robot", "none"]
    # detection_method = ["FINE+K", "UNICON+K"]
    # train_noise_method = ["unicon", "none", "ours"]
    # num_model = [1]

    ## grid search parameters CP
    estimation_method =  ["total_variation", "none"]
    detection_method = [ "FINE+K",]
    train_noise_method = ["none", "unicon"] #["ours"]
    num_model = [1]

    ## a string, or a list of strings with the same length as launch_jobs
    ## this is the names of the jobs on SCC, can be anything

    ## set SCC config, some other options can be found in "scripts" folder.
    script_path = "scripts/clothing_2GPU_template.sh"
    ################################################# No need to change the rest
    # make a grid of all possible combinations
    combinations = list(
        itertools.product(estimation_method, detection_method, train_noise_method, num_model)
    )

    launch_jobs = []
    base_names = []

    for command in launch_jobs_tmp:
        ## append the command from combinations
        for combination in combinations:
            new_command = f"\n{command} --estimation_method {combination[0]} --detection_method {combination[1]} --train_noise_method {combination[2]} --num_model {combination[3]} --seed {random.randint(100, 100000)}"
            launch_jobs.append(new_command)
            base_names.append(
                f"{combination[0]}_{combination[1]}_{combination[2]}_{combination[3]}"
            )
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int)

    args = parser.parse_args()

    ## clean up and make a tmp folder
    tmp_folder = "__tmp_scripts"
    try:
        os.system(f"rm -rf {tmp_folder}")
    except Exception as e:
        print(f"exception while rm -rf {tmp_folder}: {e}")
    mkdir(tmp_folder)

    new_scripts = []
    if not isinstance(base_names, list):
        base_names = [base_names] * len(launch_jobs)
    else:
        base_names = base_names
    ## loop through all launch commands

    for i, job in enumerate(launch_jobs):
        ## create a script files
        new_script_path = os.path.join(tmp_folder, f"{base_names[i]}_{i + 1}.sh")
        os.system(f"cp {script_path} {new_script_path}")

        ## auto add SCC job id to the launch command
        if "sccid=$JOB_ID" not in job:
            job += " --sccid=$JOB_ID"
        with open(new_script_path, "a") as f:
            f.write(f"{job}")
        new_scripts.append(new_script_path)

    for new_script_path in new_scripts:
        os.system(f"qsub {new_script_path}")

    print(f"submitted {len(new_scripts)} jobs!")
    current_user = getpass.getuser()
    os.system(f"watch -n 2 qstat -u {current_user}")
