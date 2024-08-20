import argparse
import yaml
import subprocess
import os
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='Update YAML for a sweep and provide run command')
    parser.add_argument('--experiment-name', type=str, default="classification_experiment_dev", help='The name of the experiment')
    parser.add_argument('--fc', type=str, default="geneformer", help='F_c to use')
    parser.add_argument('--fg', type=str, default="gene_id", help='F_g to use')
    parser.add_argument('--project-name', type=str, default="test", help='')

    args = parser.parse_args()
    return args

def update_yaml(args):
    # Load the YAML file
    with open("sweeps/base_sweep.yaml", 'r') as file:
        data = yaml.safe_load(file)
    
    # Update the data with arguments
    data['command'].append(f"f_c={args.fc}")
    data['command'].append(f"f_g={args.fg}")
    data['command'].append(f"+experiments={args.experiment_name}")

    # Write the modified data to a different file
    with open("sweeps/tmp.yaml", 'w') as file:
        yaml.safe_dump(data, file)
    return data

def create_and_run_sweep(args):
    create_sweep = f"wandb sweep --entity Heimdall --project {args.project_name} sweeps/tmp.yaml"
    sweep_id = subprocess.run(create_sweep, shell=True, capture_output=True)
    out = sweep_id.stderr.decode('utf-8').strip()
    wandb_agent_command = out.split("Run sweep agent with: ")[-1]
    print(wandb_agent_command)


def main():
    args = parse_arguments()
    data = update_yaml(args)
    create_and_run_sweep(args)

if __name__ == '__main__':
    main()

