from baseline.AmpleGCG.llm import AmpleLLM, LocalVLLM


def load_target_models_amplegcg(args):
    test_generation_kwargs = {
        "max_tokens": args.target_max_n_tokens,
        "temperature": 0.01,
    }
    target_llm = LocalVLLM(
        args=args,
        test_generation_kwargs=test_generation_kwargs,
        suffix_dict=args.suffix_dict,
    )
    return target_llm
