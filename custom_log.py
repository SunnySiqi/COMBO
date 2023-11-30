"""
Add your Wandb key to .env file:

    `echo WANDB_API_KEY=your_wandb_key >> .env`

"""
from typing import Union, Dict, Optional

import time
import os
from argparse import Namespace
import wandb
from dotenv import load_dotenv


def get_machine_name():
    import socket

    machine_name = socket.gethostname()
    return machine_name


def exists(val):
    return val is not None


def default(val, default):
    return val if exists(val) else default


def init_wandb(args: Namespace, project_name: str, sccid: Optional[str]):
    """
    job_id: SCC job id
    """
    # wandb.run.dir
    # https://docs.wandb.ai/guides/track/advanced/save-restore

    try:
        load_dotenv()
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        wandb.login(key=os.getenv("WANDB_API_KEY"))
    except Exception as e:
        print(f"--- was trying to log in Weights and Biases... e={e}")

    ## run_name for wandb's run
    machine_name = get_machine_name()
    watermark = f"{args.traintools}__estimation-{args.estimation_method}__detection-{args.detection_method}__train_noise-{args.train_noise_method}__nummodel-{args.num_model}_sccid{args.sccid}_cuda{args.device}_seed{args.seed}_noiseratio{args.noise_ratio}_noisetype{args.noise_type}_{machine_name}_{time.strftime('%I-%M%p-%B-%d-%Y')}"

    wandb.init(
        project=project_name,
        entity="chammi",
        name=watermark,
        settings=wandb.Settings(start_method="fork"),
    )

    return watermark


class MyLogging:
    def __init__(self, config: Dict, project_name: str, dataset: str):
        self._config = config
        self.args = config.__dict__["args"]
        self.use_wandb = self.args.no_wandb  # I know, naming here is quite confusing :)
        self.dataset = dataset
        if self.use_wandb:
            init_wandb(args=self.args, project_name=project_name, sccid=self.args.sccid)
            self.log_config()

        self.original_logger = config.get_logger("trainer", config["trainer"]["verbosity"])

    def log(
        self,
        msg: Union[Dict, str],
        use_wandb: Optional[bool] = None,
        sep=", ",
        padding_space=False,
        pref_msg: str = "",
    ):
        use_wandb = default(use_wandb, self.use_wandb)

        if isinstance(msg, Dict):
            msg_str = (
                pref_msg
                + " "
                + sep.join(
                    f"{k} {round(v, 4) if isinstance(v, int) else v}" for k, v in msg.items()
                )
            )

            ## flatten values in msg
            msg_dict = {}
            for k, v in msg.items():
                if isinstance(v, list):
                    if len(v) == 1:
                        msg_dict[k] = v[0]
                    else:
                        for i, v_i in enumerate(v):
                            msg_dict[f"{k}_{i}"] = v_i
                else:
                    msg_dict[k] = v
            if padding_space:
                msg_str = sep + msg_str + " " + sep

            if use_wandb:
                wandb.log(msg_dict)
            print(msg_str)
        else:
            print(msg)

    def warning(self, *args, **kwargs):
        self.original_logger.warning(*args, **kwargs)

    def info(self, *args, **kwargs):
        self.original_logger.info(*args, **kwargs)

    def log_config(self):
        config_dict = vars(self.args)
        config_dict.update({"dataset": self.dataset})
        config_dict.update({"every": self._config["detection"]["every"]})
        config_dict.update({"warmup": self._config["trainer"]["warmup"]})
        wandb.config.update(config_dict)  # , allow_val_change=True)

    def finish(
        self,
        use_wandb: Optional[bool] = None,
        msg_str: Optional[str] = None,
    ):
        use_wandb = default(use_wandb, self.use_wandb)

        if exists(msg_str):
            self.original_logger.info(msg_str)
        if use_wandb:
            wandb.finish()
