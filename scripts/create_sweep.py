import argparse
import subprocess

import yaml


def parse_arguments():
    parser = argparse.ArgumentParser(description="Update YAML for a sweep and provide run command")
    parser.add_argument("--commands", type=str, required=True, help="Commands as a single space-separated string")
    parser.add_argument("--project-name", type=str, default="test", help="The project name in WandB")
    parser.add_argument("--precision", type=str, default="bf16", help="precision")

    args = parser.parse_args()
    return args


def update_yaml(args):
    # Load the YAML file

    if args.precision == "bf16":
        with open("sweeps/base_sweep.yaml") as file:
            data = yaml.safe_load(file)
    else:
        with open("sweeps/fp_base_sweep.yaml") as file:
            data = yaml.safe_load(file)

    # Split the commands by spaces and add each as an entry in data['command']
    commands = args.commands.split()
    for command in commands:
        data["command"].append(command)
    # data['command'].append(f"+experiments={args.experiment_name}")

    # Write the modified data to a temporary YAML file
    with open("sweeps/tmp.yaml", "w") as file:
        yaml.safe_dump(data, file)
    return data


def create_and_run_sweep(args):
    create_sweep = f"wandb sweep --entity Heimdall --project {args.project_name} sweeps/tmp.yaml"
    sweep_id = subprocess.run(create_sweep, shell=True, capture_output=True)
    out = sweep_id.stderr.decode("utf-8").strip()
    wandb_agent_command = out.split("Run sweep agent with: ")[-1]
    print(wandb_agent_command)


def main():
    args = parse_arguments()
    update_yaml(args)
    create_and_run_sweep(args)


if __name__ == "__main__":
    main()
