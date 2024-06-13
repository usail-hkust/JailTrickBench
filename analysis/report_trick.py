import time
import os
import json
import argparse
from prettytable.colortable import ColorTable, Themes
import re
from utils.test_utils import attack_rename, defense_rename, re_calculate_asr


def short_str(s, max_len=40):
    if len(s) > max_len:
        return s[:20] + "..." + s[-5:]
    return s


path_prefix = "./exp_results/"

data_dict = {}


def read_exp_results(exp_list, path_prefix):
    for exp in exp_list:
        exp_path = os.path.join(path_prefix, exp)
        print("exp_path: ", exp_path)
        data_dict[exp] = {}
        if not os.path.exists(exp_path):
            print("Path not exist: ", exp_path)
            continue
            # raise ValueError("Path not exist")
        dataset_list = sorted(os.listdir(exp_path))
        for dataset in dataset_list:
            dataset_path = os.path.join(exp_path, dataset)
            print("dataset_path: ", dataset_path)
            data_dict[exp][dataset] = {}
            defense_list = sorted(os.listdir(dataset_path))
            for defense in defense_list:
                defense_path = os.path.join(dataset_path, defense)
                print("defense_path: ", defense_path)
                data_dict[exp][dataset][defense] = {}
                attack_list = sorted(os.listdir(defense_path))
                for attack in attack_list:
                    attack_path = os.path.join(defense_path, attack)
                    print("attack_path: ", attack_path)
                    data_dict[exp][dataset][defense][attack] = {}
                    c_exp_name_list = sorted(os.listdir(attack_path))
                    for c_exp_name in c_exp_name_list:
                        # c_exp_name = os.listdir(attack_path)[0]
                        file_path = os.path.join(attack_path, c_exp_name)
                        file_list = os.listdir(file_path)
                        # check the length of the file list
                        if len(file_list) != 1:
                            print("file_list: ", file_list)
                            raise ValueError("The length of the file list is not 1")

                        # check if the file is a json file
                        if file_list[0].endswith(".json"):
                            with open(os.path.join(file_path, file_list[0]), "r") as f:
                                curr_file = json.load(f)
                                # check the length of the file
                                if (
                                    len(curr_file) != 100
                                    and len(curr_file) != 50
                                    and len(curr_file) != 30
                                ):
                                    print("Len", len(curr_file))
                                    raise ValueError(
                                        "The length of the file is not 100 or 50"
                                    )
                            # print(curr_file)
                            # print("=" * 30)
                        else:
                            print("file_list: ", file_list)
                            raise ValueError("The file is not a json file")
                        c_defense_name, c_model_name, c_attack_name, c_timestamp = (
                            file_list[0].split("__")
                        )
                        c_defense_name = c_defense_name[8:]
                        c_attack_name = c_attack_name.split("_")[-1]
                        c_timestamp = c_timestamp.split(".")[0][2:]
                        print(c_defense_name, c_model_name, c_attack_name, c_timestamp)
                        # check defense and attack name
                        if (
                            defense_rename[c_defense_name] != defense
                            or attack_rename[c_attack_name] != attack
                        ):
                            print("Defense: ", defense_rename[c_defense_name], defense)
                            print("Attack: ", attack_rename[c_attack_name], attack)
                            raise ValueError("Defense or attack name not match")
                        c_avg_iter = round(
                            sum([item["attack_iterations"] for item in curr_file])
                            / len(curr_file),
                            0,
                        )
                        curr_asr = round(
                            sum([item["is_JB"] for item in curr_file])
                            / len(curr_file)
                            * 100,
                            2,
                        )
                        curr_asr_agent_list = [
                            item["is_JB_Agent"] for item in curr_file
                        ]
                        if "None" in curr_asr_agent_list or None in curr_asr_agent_list:
                            raise ValueError("None in curr_asr_agent_list")
                        else:
                            curr_asr_agent = round(
                                sum(curr_asr_agent_list) / len(curr_file) * 100, 2
                            )
                        curr_file = re_calculate_asr(curr_file)
                        curr_asr_recal = round(
                            sum([item["is_JB_recal"] for item in curr_file])
                            / len(curr_file)
                            * 100,
                            2,
                        )
                        asr_diff = round(curr_asr - curr_asr_recal, 2)
                        data_dict[exp][dataset][defense][attack][c_exp_name] = [
                            exp,
                            dataset,
                            defense,
                            attack,
                            c_model_name,
                            c_exp_name,
                            c_timestamp,
                            c_avg_iter,
                            curr_asr,
                            curr_asr_agent,
                            curr_asr_recal,
                            asr_diff,
                        ]

    return


def filter_Null(s):
    if s == None:
        return 0
    return s


def check_progress(exp_list, path_prefix, sort_by="Exp"):
    table = ColorTable(theme=Themes.OCEAN)
    table.field_names = [
        "Exp",
        "Data",
        "Defense",
        "Attack",
        "Model",
        "Name",
        "Timestamp",
        "#Iters.",
        "ASR (%)",
        "Agent ASR (%)",
        "ASR_recal (%)",
        "Diff(%)",
    ]
    read_exp_results(exp_list, path_prefix)
    print("=" * 30)
    # add rows for current dataset from data_dict
    order = [
        "5GCG",
        "1AutoDAN",
        "7AmpleGCG",
        "6AdvPrompter",
        "2PAIR",
        "3TAP",
        "4GPTFuzz",
    ]
    for exp in data_dict:
        if data_dict[exp]:
            for dataset in data_dict[exp]:
                for defense in data_dict[exp][dataset]:
                    for attack in data_dict[exp][dataset][defense]:
                        for c_exp_name in data_dict[exp][dataset][defense][attack]:
                            table.add_row(
                                data_dict[exp][dataset][defense][attack][c_exp_name]
                            )
        else:
            table.add_row([exp, "NA", "NA", "NA", "NA", "NA", "-", -1, -1, -1, -1, -1])

    if sort_by == "Exp":
        table.sortby = "Exp"
    elif sort_by == "Data":
        table.sortby = "Data"
    elif sort_by == "Defense":
        table.sortby = "Defense"
    elif sort_by == "Model":
        table.sortby = "Model"
    elif sort_by == "Attack":
        table.sortby = "Attack"
    else:
        table.sortby = "Exp"
    print(table)


if __name__ == "__main__":
    # read arguments
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--exp",
        type=str,
        default="debug",
        choices=["ml", "mv", "t", "a", "debug", "wb"],
        help="The path to save the results",
    )

    args = parser.parse_args()
    main_llama = [
        "main_llama",
    ]
    main_vicuna = [
        "main_vicuna",
    ]
    target_exp = [
        "trick_target_align_autodan",
        "trick_target_align_pair",
        "trick_target_size_autodan",
        "trick_target_size_pair",
        "trick_target_system_autodan",
        "trick_target_system_pair",
        "trick_target_system_autodan",
        "trick_target_system_pair",
        "trick_target_template_autodan",
        "trick_target_template_pair",
    ]
    atk_exp = [
        "trick_atk_ability_pair",
        "trick_atk_budget_gcg",
        "trick_atk_budget_pair",
        "trick_atk_suffix_length",
        "trick_atk_intension_autodan",
        "trick_atk_intension_pair",
    ]
    debug_exp = [
        "trick_atk_suffix_length",
    ]
    exp_name_dict = {
        "ml": main_llama,
        "mv": main_vicuna,
        "t": target_exp,
        "a": atk_exp,
        "debug": debug_exp,
    }
    check_progress(
        exp_list=exp_name_dict[args.exp], path_prefix=path_prefix, sort_by="Exp"
    )
    print("\n" + "=" * 50 + "\n")
    print(exp_name_dict[args.exp])
