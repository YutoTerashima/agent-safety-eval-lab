from agent_safety_eval_lab.mock_runner import run_demo


if __name__ == "__main__":
    summary, lines = run_demo()
    print(" ".join(f"{key}={value}" for key, value in summary.items()))
    for line in lines:
        print(line)
