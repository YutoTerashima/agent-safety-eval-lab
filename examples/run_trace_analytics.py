from agent_safety_eval_lab.experiment import write_experiment_outputs


if __name__ == "__main__":
    for path in write_experiment_outputs():
        print(path)
