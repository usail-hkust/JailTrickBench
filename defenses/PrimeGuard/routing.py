from utils import route_templates, extract_and_eval_json

def route(prompt, model, env, cached_results):
    """
    Process the input prompt and return the routing results.

    Args:
        prompt (str): The input prompt to process.
        model (LitellmModel): The model instance for making API calls.
        env (jinja2.Environment): The Jinja2 environment for template rendering.
        cached_results (list): List of cached results.

    Returns:
        dict: A dictionary containing the routing results.
    """
    cached_result = next(
        (item for item in cached_results if item["prompt"] == prompt), None
    )

    if cached_result:
        return cached_result

    # Vanilla Approach
    system_instructions = env.get_template("oai_safety.j2").render()
    vanilla_result = model.batch_call([prompt], system_prompt=system_instructions)[0]

    # PrimeGuard Approach
    routing_template = env.get_template("route_selection.j2")
    routing_rendered = routing_template.render(
        system_prompt=system_instructions, user_input=prompt
    )
    route_selection_output = model.batch_call([routing_rendered])[0]

    final_conv, routes, system_check_results, system_tips = route_templates(
        route_selection_outputs=[route_selection_output],
        prompts=[prompt],
        restrictive_system_instructions=system_instructions,
        env=env,
        no_violation_template_name="answer_utility.j2",
        potential_violation_template_name_first_turn="display_analysis.j2",
        potential_violation_template_name_second_turn="get_answer.j2",
        direct_violation_template_name="refusal.j2",
    )

    final_output = model.batch_call(final_conv)[0]

    primeguard_answer = final_output
    reevaluation = "N/A"
    if routes[0] == "potential_violation":
        parsed_json = extract_and_eval_json(final_output)
        if len(parsed_json) > 0:
            if (
                "reevaluation" in parsed_json[0].keys()
                and "final_response" in parsed_json[0].keys()
            ):
                reevaluation = parsed_json[0]["reevaluation"]
                primeguard_answer = parsed_json[0]["final_response"]

    return {
        "route": routes[0],
        "vanilla_result": vanilla_result,
        "primeguard_result": primeguard_answer,
        "system_check": system_check_results[0],
        "system_tip": system_tips[0],
        "reevaluation": reevaluation,
    }
