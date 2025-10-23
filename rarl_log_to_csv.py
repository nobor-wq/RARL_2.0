import csv


# 定义读取txt文件并提取信息的函数
def extract_parameters_results_from_txt(txt_file_path, csv_file_path):
    # 读取txt文件的内容
    with open(txt_file_path, 'r') as file:
        txt_data = file.read()

    # 分割每个 section
    sections = txt_data.split("--------------------------------------------------")
    data = []

    # 从每个section提取信息
    for section in sections:
        if "env_name" in section and "Results" in section:
            try:
                env_name = section.split("env_name: ")[1].split("\n")[0]
                adv_steps = section.split("adv_steps: ")[1].split("\n")[0]
                train_step = section.split("train_step: ")[1].split("\n")[0]
                addition_msg = section.split("addition_msg: ")[1].split("\n")[0]
                attack_eps = section.split("attack_eps: ")[1].split("\n")[0]
                attack = section.split("attack: ")[1].split("\n")[0]
                adv_iteration = section.split("adv_iteration: ")[1].split("\n")[0]
                agent_iteration = section.split("agent_iteration: ")[1].split("\n")[0]
                collision_rate = section.split("Collision rate: ")[1].split("\n")[0]
                success_rate = section.split("Success rate: ")[1].split("\n")[0]
                mean_reward = section.split("Mean reward: ")[1].split("\n")[0]
                mean_steps = section.split("Mean steps: ")[1].split("\n")[0]
                mean_speed = section.split("Mean speed: ")[1].split("\n")[0]
                mean_attack_times = section.split("Mean attack times: ")[1].split("\n")[0]
                success_attack_rate = section.split("Success attack rate: ")[1].split("\n")[0]
                mean_attack_reward = section.split("Reward per attack: ")[1].split("\n")[0]


                # 将提取的信息存入列表
                # data.append([env_name, adv_algo, adv_steps, mean_reward, mean_steps, mean_speed, collision_rate, success_rate, mean_attack_times])
                data.append(
                    [env_name, adv_steps, train_step, addition_msg, attack_eps, attack, adv_iteration, agent_iteration, collision_rate,
                     success_rate, mean_reward, mean_steps, mean_speed, mean_attack_times, success_attack_rate,
                     mean_attack_reward])
            except Exception as e:
                raise RuntimeError(f"解析 section 时发生错误: {e}") from e

    # 将提取的数据写入csv文件
    # header = ["env_name", "adv_algo", "adv_steps", "mean_reward", "mean_steps", "mean_speed", "collision_rate", "success_rate", "mean_attack_times"]
    header = ["env_name", "adv_steps", "train_step", "addition_msg", "attack_eps", "attack", "adv_iteration", "agent_iteration", "collision_rate",
              "success_rate", "mean_reward", "mean_steps", "mean_speed", "mean_attack_times", "success attack rate",
              "mean_attack_reward"]
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)


# 使用示例：提取txt文件内容并保存为csv
txt_file_path = "rarl_eval_log.txt"  # 替换为实际txt文件路径
csv_file_path = "rarl_eval_log.csv"  # 替换为保存csv文件的路径

extract_parameters_results_from_txt(txt_file_path, csv_file_path)
