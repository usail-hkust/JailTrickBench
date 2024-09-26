import numpy as np
from tqdm import tqdm
import datetime


from utils.utils import load_model_and_tokenizer, set_random_seed
from utils.string_utils import load_prompts, load_goals
from utils.test_utils import (
    get_template_name,
    save_test_to_file,
    test_prefixes,
    load_test_from_file,
    load_test_from_file_split,
    save_test_to_file_split,
    load_split_file_whole,
    instruction2dratk_data_path,
)

# import args
from initialize_args import initialize_args

# import attack baselines
from baseline.GCG.GCG_single_main import GCG
from baseline.AutoDAN.AutoDAN_single_main import AutoDAN_single_main
from baseline.TAP.TAP_single_main import TAP_single_main, TAP_initial
from baseline.PAIR.PAIR_single_main import PAIR_single_main, PAIR_initial
from baseline.GPTFuzz.GPTFuzz_single_main import GPTFuzz_initial, GPTFuzz_single_main
from baseline.AmpleGCG.AmpleGCG_single_main import (
    AmpleGCG_initial,
    AmpleGCG_single_main,
    AmpleGCG_generate_suffix,
)
from baseline.AdvPrompter.AdvPrompter_single_main import (
    AdvPrompter_initial,
    AdvPrompter_generate_suffix,
    AdvPrompter_single_main,
)
from baseline.AmpleGCG.utils import load_target_models_amplegcg
from baseline.AdvPrompter.utils import load_target_models_advprompter
from baseline.DrAttack.DrAttack_single_main import DrAttack_initial, DrAttack_single_main, DrAttack_stop
from baseline.MultiJail.MultiJail_single_main import (
    MultiJail_initial,
    MultiJail_generate_suffix,
    MultiJail_single_main,
)
from baseline.MultiJail.utils import load_target_models_MultiJail
# import defense methods
from defense import test_smoothLLM, generate_defense_goal

# import evaluation agent
from GPTEvaluatorAgent.agent_eval import agent_evaluation


def generate_attack_result(goal, target, models, device, args, curr_output):

    if args.attack == "GCG":
        model, tokenizer = models[0], models[1]
        curr_args_dict = vars(args)
        adv_prompt, model_output, iteration, is_JB = GCG(
            model=model,
            tokenizer=tokenizer,
            device=device,
            goal=goal,
            target=target,
            args_dict=curr_args_dict,
        )
        curr_output["adv_prompt"] = adv_prompt
        curr_output["language_model_output"] = model_output
        curr_output["attack_iterations"] = iteration
        curr_output["is_JB"] = is_JB
    elif args.attack == "AutoDAN":
        model, tokenizer = models[0], models[1]
        curr_args_dict = vars(args)
        adv_prompt, model_output, iteration, is_JB = AutoDAN_single_main(
            args_dict=curr_args_dict,
            target_model=model,
            target_tokenizer=tokenizer,
            goal=goal,
            target=target,
        )
        curr_output["adv_prompt"] = adv_prompt
        curr_output["language_model_output"] = model_output
        curr_output["attack_iterations"] = iteration
        curr_output["is_JB"] = is_JB
    elif args.attack == "AmpleGCG":
        model = models[0]
        curr_args_dict = vars(args)
        adv_prompt, model_output, iteration, is_JB = AmpleGCG_single_main(
            args_dict=curr_args_dict,
            target_model=model,
            goal=goal,
            target=target,
        )
        curr_output["adv_prompt"] = adv_prompt
        curr_output["language_model_output"] = model_output
        curr_output["attack_iterations"] = iteration
        curr_output["is_JB"] = is_JB
    elif args.attack == "AdvPrompter":
        model = models[0]
        curr_args_dict = vars(args)
        adv_prompt, model_output, iteration, is_JB = AdvPrompter_single_main(
            args_dict=curr_args_dict,
            target_model=model,
            goal=goal,
            target=target,
        )
        curr_output["adv_prompt"] = adv_prompt
        curr_output["language_model_output"] = model_output
        curr_output["attack_iterations"] = iteration
        curr_output["is_JB"] = is_JB
    elif args.attack == "DrAttack":
        curr_args_dict = vars(args)
        curr_output_record = DrAttack_single_main(
            args_dict=curr_args_dict,
            worker=models[0],
            attack=models[1],
            goal=goal,
        )
        curr_output["adv_prompt"] = curr_output_record["adv_prompt"]
        curr_output["optimized_sentence"] = curr_output_record["optimized_sentence"]
        curr_output["language_model_output"] = curr_output_record["language_model_output"]
        curr_output["negative_similarity_score"] = curr_output_record["negative_similarity_score"]
        curr_output["attack_iterations"] = curr_output_record["attack_iterations"]
        curr_output["is_JB"] = curr_output_record["is_JB"]
    elif args.attack == "MultiJail":
        model = models[0]
        curr_args_dict = vars(args)
        adv_prompt, model_output, iteration, is_JB = MultiJail_single_main(
            args_dict=curr_args_dict,
            target_model=model,
            goal=goal,
            target=target,
            curr_output=curr_output,
        )
        curr_output["adv_prompt"] = adv_prompt
        curr_output["language_model_output"] = model_output
        curr_output["attack_iterations"] = iteration
        curr_output["is_JB"] = is_JB
    elif args.attack == "TAP":
        attack_llm, target_llm, evaluator_llm = models[0], models[1], models[2]
        curr_args_dict = vars(args)
        curr_output_record = TAP_single_main(
            args_dict=curr_args_dict,
            attack_llm=attack_llm,
            target_llm=target_llm,
            evaluator_llm=evaluator_llm,
            goal=goal,
            target=target,
        )
        curr_output["language_model_output"] = curr_output_record[
            "language_model_output"
        ]
        curr_output["attack_iterations"] = curr_output_record["attack_iterations"]
        curr_output["is_JB"] = curr_output_record["is_JB"]
        curr_output["is_JB_Judge"] = curr_output_record["is_JB_Judge"]
        curr_output["attack_prompt"] = curr_output_record["attack_prompt"]
        curr_output["improve_prompt"] = curr_output_record["improve_prompt"]
        curr_output["judge_output"] = curr_output_record["judge_output"]
        curr_output["on_topic_score"] = curr_output_record["on_topic_score"]
    elif args.attack == "PAIR":
        attack_llm, target_llm, evaluator_llm = models[0], models[1], models[2]
        curr_args_dict = vars(args)

        curr_output_record = PAIR_single_main(
            args_dict=curr_args_dict,
            attackLM=attack_llm,
            targetLM=target_llm,
            judgeLM=evaluator_llm,
            goal=goal,
            target=target,
        )
        curr_output["language_model_output"] = curr_output_record[
            "language_model_output"
        ]
        curr_output["attack_iterations"] = curr_output_record["attack_iterations"]
        curr_output["is_JB"] = curr_output_record["is_JB"]
        curr_output["is_JB_Judge"] = curr_output_record["is_JB_Judge"]
        curr_output["attack_prompt"] = curr_output_record["attack_prompt"]
        curr_output["improve_prompt"] = curr_output_record["improve_prompt"]
        curr_output["judge_output"] = curr_output_record["judge_output"]
    elif args.attack == "GPTFuzz":
        openai_model, target_model, roberta_model = models[0], models[1], models[2]
        curr_args_dict = vars(args)

        curr_output_record = GPTFuzz_single_main(
            args_dict=curr_args_dict,
            openai_model=openai_model,
            target_model=target_model,
            roberta_model=roberta_model,
            goal=goal,
            target=target,
        )
        curr_output["language_model_output"] = curr_output_record[
            "language_model_output"
        ]
        curr_output["attack_iterations"] = curr_output_record["attack_iterations"]
        curr_output["is_JB"] = curr_output_record["is_JB"]
        curr_output["is_JB_Judge"] = curr_output_record["is_JB_Judge"]
        curr_output["attack_prompt"] = curr_output_record["attack_prompt"]
    else:
        raise NameError
    return curr_output


def test(goals, targets, models, device, args, all_output=[]):
    if args.attack == "DrAttack" and args.defense_type in ["self_reminder", "RPO", "smoothLLM"]:
            instruction_name = args.instructions_path.split("/")[-1]
            pert_goals_path = instruction2dratk_data_path[instruction_name][args.defense_type]
            # pert_goals = load_pert_goals(pert_goals_path)
            pert_goals = load_goals(pert_goals_path)
    else:
        pert_goals = [
            generate_defense_goal(
                goal_i,
                defense_type=args.defense_type,
                pert_type=args.pert_type,
                smoothllm_pert_pct=args.smoothllm_pert_pct,
            )
            for goal_i in goals
        ]
    if args.attack == "AmpleGCG":
        args.suffix_dict = AmpleGCG_generate_suffix(args, pert_goals)
        models = [load_target_models_amplegcg(args)]
    elif args.attack == "AdvPrompter":
        args.suffix_dict = AdvPrompter_generate_suffix(args, pert_goals)
        models = [load_target_models_advprompter(args)]
    elif args.attack == "MultiJail":
        args.suffix_dict = []
        models = [load_target_models_MultiJail(args)]

    for goal_i, target_i, pert_goal_i in tqdm(
        zip(
            goals[args.test_data_idx : args.end_index],
            targets[args.test_data_idx : args.end_index],
            pert_goals[args.test_data_idx : args.end_index],
        ),
        desc="Testing",
    ):
        print(f"""\n{'=' * 36}\nDefense Method: {args.defense_type}\n{'=' * 36}\n""")
        curr_output = {
            "original_prompt": goal_i,
            "perturbed_prompt": pert_goal_i,
            "target": target_i,
            "adv_prompt": "NULL",
            "language_model_output": "NULL",
            "attack_iterations": None,
            "data_id": args.test_data_idx,
            "is_JB": "None",
            "is_JB_Judge": "None",
            "is_JB_Agent": "None",
        }
        print(curr_output)

        print(f"""\n{'=' * 36}\nAttack Method: {args.attack}\n{'=' * 36}\n""")
        curr_output = generate_attack_result(
            pert_goal_i, target_i, models, device, args, curr_output
        )
        print(f"""\n{'=' * 36}\nFinish testing data_id: {args.test_data_idx}\n""")
        print(curr_output)
        print(f"""{'=' * 36}\n""")
        all_output.append(curr_output)
        if args.data_split:
            save_test_to_file_split(args=args, instruction=curr_output)
        else:
            save_test_to_file(args=args, instructions=all_output)
        args.test_data_idx += 1

    return all_output


def run(goals, targets, target_model_path, device, args, all_output=[]):
    # load models
    if args.attack in ["GCG", "AutoDAN"]:
        target_model, target_tokenizer = load_model_and_tokenizer(
            target_model_path, tokenizer_path=None, device=device
        )
        models = [target_model, target_tokenizer]
    elif args.attack == "TAP":
        args_dict = vars(args)
        args, attack_llm, target_llm, evaluator_llm = TAP_initial(args_dict=args_dict)
        models = [attack_llm, target_llm, evaluator_llm]
    elif args.attack == "PAIR":
        args_dict = vars(args)
        args, attack_llm, target_llm, evaluator_llm = PAIR_initial(args_dict=args_dict)
        models = [attack_llm, target_llm, evaluator_llm]
    elif args.attack == "GPTFuzz":
        args_dict = vars(args)
        args, openai_model, target_model, roberta_model = GPTFuzz_initial(
            args_dict=args_dict
        )
        models = [openai_model, target_model, roberta_model]
    elif args.attack == "wild_adv_prompt":
        models = []
    elif args.attack == "AmpleGCG":
        args_dict = vars(args)
        args = AmpleGCG_initial(args_dict=args_dict)
        models = []
    elif args.attack == "AdvPrompter":
        args_dict = vars(args)
        args = AdvPrompter_initial(args_dict=args_dict)
        models = []
    elif args.attack == "DrAttack":
        args_dict = vars(args)
        args, worker, attack = DrAttack_initial(args_dict=args_dict)
        models = [worker, attack]
    elif args.attack == "MultiJail":
        args_dict = vars(args)
        args = MultiJail_initial(args_dict=args_dict)
        models = []
    else:
        raise NameError
    all_output = test(goals, targets, models, device, args, all_output=all_output)
    if args.attack == "DrAttack":
        DrAttack_stop(worker=worker)
    return all_output


def main(args):

    # default setting
    set_random_seed(args.random_seed)
    target_model_path = args.target_model_path
    args.template_name = get_template_name(target_model_path)
    args.timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M_%S")
    print("\n\ntarget_model_path", target_model_path, "\n\n")
    device = "cuda:{}".format(args.device_id)
    instructions_path = args.instructions_path
    goals, targets = load_prompts(instructions_path)

    if args.data_split:
        print("Find data_split is True, split the data")
        args.start_index = (
            len(goals) // args.data_split_total_num
        ) * args.data_split_idx
        args.end_index = (len(goals) // args.data_split_total_num) * (
            args.data_split_idx + 1
        )
    else:
        args.start_index = 0
        args.end_index = len(goals)

    # test
    all_output = []
    args.test_data_idx = max(args.start_index, 0)
    # try to load data if resume the experiment
    if args.resume_exp:
        if args.data_split:
            new_start_idx, new_timestamp = load_test_from_file_split(args)
            args.test_data_idx = new_start_idx
            if len(new_timestamp) > 0:
                args.timestamp = new_timestamp
            print(
                f"Load the progress successfully, start from the index: {args.test_data_idx}; Current timestamp: {args.timestamp}"
            )
        else:
            all_output, new_timestamp = load_test_from_file(args)
            if len(all_output) == 0:
                print("Load the data failed, start from the beginning")
                print(f"Start from the index: {args.test_data_idx}")
            else:
                args.test_data_idx = all_output[-1]["data_id"] + 1
                if len(new_timestamp) > 0:
                    args.timestamp = new_timestamp
                print(
                    f"Load the data successfully, start from the index: {args.test_data_idx}; Current timestamp: {args.timestamp}"
                )
    all_output = run(goals, targets, target_model_path, device, args, all_output)

    # test smoothLLM
    if args.defense_type == "smoothLLM" and args.attack != "DrAttack":
        final_all_output = test_smoothLLM(all_output, args)
    else:
        print(f"""\n{'=' * 36}\nNo SmoothLLM Test\n{'=' * 36}\n""")
        final_all_output = all_output

    # agent evaluation
    if args.agent_evaluation:
        final_all_output = load_split_file_whole(args)
        if len(final_all_output) != len(goals):
            print(
                "Find the final_all_output is not equal to the goals, skip the agent evaluation"
            )
            return
        if not args.agent_recheck and args.resume_exp:
            print("Find resume_exp is True, check whether need to do agent evaluation")
            if final_all_output[-1]["is_JB_Agent"] != "None":
                print(f"""\n{'*' * 36}\nSkip the agent evaluation\n{'*' * 36}\n""")
                save_test_to_file(args=args, instructions=final_all_output)
                return
            else:
                print("Start the agent evaluation")
        elif args.agent_recheck:
            print("Find agent_recheck is True, start the agent evaluation")
        else:
            print("Start the agent evaluation")
        print(f"""\n{'=' * 36}\nAgent Evaluation\n{'=' * 36}\n""")
        final_all_output = agent_evaluation(args=args, data=final_all_output)
        save_test_to_file(args=args, instructions=final_all_output)
        print(f"""\n{'=' * 36}\nFinish Agent Evaluation\n{'=' * 36}\n""")


if __name__ == "__main__":
    args = initialize_args()
    # print args
    args_dict = vars(args)
    print(args_dict)
    main(args)
