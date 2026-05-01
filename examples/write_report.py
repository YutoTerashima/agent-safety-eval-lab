from agent_safety_eval_lab.report import write_report


if __name__ == "__main__":
    json_path, md_path = write_report()
    print(json_path)
    print(md_path)
