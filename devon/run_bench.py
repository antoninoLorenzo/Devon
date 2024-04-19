

import json
import logging
import os
import re
import traceback
import yaml

from dataclasses import dataclass
from getpass import getuser
from pathlib import Path
from rich.logging import RichHandler
from simple_parsing import parse
from simple_parsing.helpers import FrozenSerializable, FlattenedAccess

from devon.agent.model import ModelArguments
from devon.agent.thread import Agent
from devon.swebenchenv.environment.swe_env import EnvironmentArguments, SWEEnv
from devon.swebenchenv.environment.utils import get_data_path_name
from swebench import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
from unidiff import PatchSet

handler = RichHandler(show_time=False, show_path=False)
handler.setLevel(logging.DEBUG)
logger = logging.getLogger("run_dev")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.propagate = False
logging.getLogger("simple_parsing").setLevel(logging.WARNING)


@dataclass(frozen=True)
class ScriptArguments(FlattenedAccess, FrozenSerializable):
    environment: EnvironmentArguments
    instance_filter: str = ".*"  # Only run instances that completely match this regex
    skip_existing: bool = True  # Skip instances with existing trajectories
    suffix: str = ""

    @property
    def run_name(self):
        """Generate a unique name for this run based on the arguments."""
        model_name = self.agent.model.model_name.replace(":", "-")
        data_stem = get_data_path_name(self.environment.data_path)
        config_stem = Path(self.agent.config_file).stem

        temp = self.agent.model.temperature
        top_p = self.agent.model.top_p

        per_instance_cost_limit = self.agent.model.per_instance_cost_limit
        install_env = self.environment.install_environment

        return (
            f"{model_name}__{data_stem}__{config_stem}__t-{temp:.2f}__p-{top_p:.2f}"
            + f"__c-{per_instance_cost_limit:.2f}__install-{int(install_env)}"
            + (f"__{self.suffix}" if self.suffix else "")
        )


def main(args: ScriptArguments):
    logger.info(f"📙 Arguments: {args.dumps_yaml()}")

    print(args.__dict__)

    env = SWEEnv(args.environment)
    agent = Agent("primary")

    for index in range(len(env.data)):
        try:
            # Reset environment
            instance_id = env.data[index]["instance_id"]
            # if should_skip(args, traj_dir, instance_id):
            #     continue
            logger.info("▶️  Beginning task " + str(index))


            try:
                observation, info = env.reset(index)
            except Exception as e:
                logger.error(f"Error resetting environment: {e}")
                continue
            if info is None:
                continue

            # Get info, patch information
            issue = getattr(env, "query", None)
            files = []
            if "patch" in env.record:
                files = "\n".join(
                    [f"- {x.path}" for x in PatchSet(env.record["patch"]).modified_files]
                )
            # Get test files, F2P tests information
            test_files = []
            if "test_patch" in env.record:
                test_patch_obj = PatchSet(env.record["test_patch"])
                test_files = "\n".join(
                    [f"- {x.path}" for x in test_patch_obj.modified_files + test_patch_obj.added_files]
                )
            tests = ""
            if "FAIL_TO_PASS" in env.record:
                tests = "\n".join([f"- {x}" for x in env.record["FAIL_TO_PASS"]])

            setup_args = {
                "issue": issue,
                "files": files,
                "test_files": test_files,
                "tests": tests
            }

            traj_dir = Path("trajectories") / Path(getuser()) / Path("_".join([agent.default_model.args.model_name, str(agent.default_model.args.temperature)])) / Path(env.record["instance_id"])
            os.makedirs(traj_dir, exist_ok=True)
            save_arguments(traj_dir, args)
            
            info = agent.run(
                setup_args=setup_args,
                env=env,
                observation=observation,
                traj_dir=traj_dir,
                return_type="info",
            )
            save_predictions(traj_dir, instance_id, info)

        except KeyboardInterrupt:
            logger.info("Exiting InterCode environment...")
            env.close()
            break
        except Exception as e:
            traceback.print_exc()
            logger.warning(f"❌ Failed on {env.record['instance_id']}: {e}")
            env.reset_container()
            continue


def save_arguments(traj_dir, args):
    """Save the arguments to a yaml file to the run's trajectory directory."""
    log_path = traj_dir / "args.yaml"

    if log_path.exists():
        try:
            other_args = args.load_yaml(log_path)
            if (args.dumps_yaml() != other_args.dumps_yaml()):  # check yaml equality instead of object equality
                logger.warning("**************************************************")
                logger.warning("Found existing args.yaml with different arguments!")
                logger.warning("**************************************************")
        except Exception as e:
            logger.warning(f"Failed to load existing args.yaml: {e}")

    with log_path.open("w") as f:
        args.dump_yaml(f)


def should_skip(args, traj_dir, instance_id):
    """Check if we should skip this instance based on the instance filter and skip_existing flag."""
    # Skip instances that don't match the instance filter
    if re.match(args.instance_filter, instance_id) is None:
        logger.info(f"Instance filter not matched. Skipping instance {instance_id}")
        return True

    # If flag is set to False, don't skip
    if not args.skip_existing:
        return False

    # Check if there's an existing trajectory for this instance
    log_path = traj_dir / (instance_id + ".traj")
    if log_path.exists():
        with log_path.open("r") as f:
            data = json.load(f)
        # If the trajectory has no exit status, it's incomplete and we will redo it
        exit_status = data["info"].get("exit_status", None)
        if exit_status == "early_exit" or exit_status is None:
            logger.info(f"Found existing trajectory with no exit status: {log_path}")
            logger.info("Removing incomplete trajectory...")
            os.remove(log_path)
        else:
            logger.info(f"⏭️ Skipping existing trajectory: {log_path}")
            return True
    return False


def save_predictions(traj_dir, instance_id, info):
    output_file = Path(traj_dir) / "all_preds.jsonl"
    model_patch = info["submission"] if "submission" in info else None
    datum = {
        KEY_MODEL: Path(traj_dir).name,
        KEY_INSTANCE_ID: instance_id,
        KEY_PREDICTION: model_patch,
    }
    with open(output_file, "a+") as fp:
        print(json.dumps(datum), file=fp, flush=True)
    logger.info(f"Saved predictions to {output_file}")


if __name__ == "__main__":

    with open ("tasklist", "r") as f:
        tasks = f.readlines()
    tasks = [x.strip() for x in tasks]

    issues = [
    "astropy__astropy-12907",
    "django__django-13230",
    "sympy__sympy-16988",
    "django__django-17051",
    "django__django-11049",
    "pytest__pytest-7373",
    "pytest__pytest-5221",
    "django__django-12700",
    "sympy__sympy-12481",
    "matplotlib__matplotlib-25079",
    "django__django-12856",
    "django__django-16229",
    "django__django-11283",
    "sympy__sympy-14817",
    "sympy__sympy-16106",
    "scikit-learn__scikit-learn-14817",
    "matplotlib__matplotlib-24334",
    "pytest__pytest-7432",
    "astropy__astropy-12907",
    "psf__requests-2674",
]

    defaults = ScriptArguments(
        suffix="",
        environment=EnvironmentArguments(
            image_name="swe-agent",
            data_path="princeton-nlp/SWE-bench_Lite",
            split="test",
            verbose=True,
            container_name="swe-agent2",
            install_environment=True,
            specific_issues=["django__django-13028"]
        ),
        skip_existing=True,
    )

    main(defaults)