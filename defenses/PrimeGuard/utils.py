import json
from typing import List, Dict, Optional
from jinja2 import Environment


def extract_and_eval_json(input_text: str, max_jsons: Optional[int] = 1) -> List[Dict]:
    """
    Extracts JSON-like substrings from a given input text and converts them into Python dictionaries.
    This function tries to handle cases where JSON objects might not be properly closed by assuming
    they end at the end of the string. It stops parsing once the specified number of JSON objects is reached.

    Parameters:
    - input_text (str): A string that potentially contains one or more JSON-like substrings.
    - max_jsons (Optional[int]): The maximum number of JSON objects to parse. Defaults to 1. If None, all JSON objects will be parsed.

    Returns:
    - list: A list of dictionaries. Each dictionary is a successfully parsed JSON object from
            the input text. If no valid JSON objects are found, or if all JSON-like substrings
            fail to parse, the list will be empty.

    Raises:
    - json.JSONDecodeError: If `json.loads()` encounters a string that is not valid JSON, this
                            exception will be caught and handled internally by the function.
                            The function will continue to parse other substrings, if any.

    Example Usage:
    --------------
    example_string = '''
    {
    "system_check_result": "The response could lead to harm.",
    "route": "potential_violation",
    "system_tip": "Decline the offer and seek help from trusted institutions."
    '''
    parsed_jsons = extract_and_eval_json(example_string, max_jsons=1)
    for parsed_json in parsed_jsons:
        print(parsed_json)

    Output:
    -------
    [{'system_check_result': 'The response could lead to harm.', 'route': 'potential_violation',
    'system_tip': 'Decline the offer and seek help from trusted institutions.'}]
    """

    input_text = input_text.replace("\n", "")
    results = []
    brace_count = 0
    start_index = None
    json_count = 0
    if max_jsons is None:
        max_jsons = float("inf")
    if max_jsons == 1:
        # directly try and parse the entire input text as a single JSON object
        try:
            parsed_dict = json.loads(input_text)
            return [parsed_dict]
        except json.JSONDecodeError:
            pass

    for index, char in enumerate(input_text):
        if char == "{":
            if brace_count == 0:
                start_index = index  # Start of a new JSON object
            brace_count += 1
        elif char == "}":
            brace_count -= 1

        # If we encounter a complete JSON object
        if brace_count == 0 and start_index is not None:
            json_candidate = input_text[start_index : index + 1]
            try:
                parsed_dict = json.loads(json_candidate)
                results.append(parsed_dict)
                json_count += 1
                if max_jsons is not None and json_count >= max_jsons:
                    return results
                start_index = None  # Reset start index after successfully parsing
            except json.JSONDecodeError:
                continue

    # Attempt to fix and parse if we have an unclosed JSON object at the end
    if brace_count != 0 and start_index is not None:
        json_candidate = input_text[start_index:] + "}"
        try:
            parsed_dict = json.loads(json_candidate)
            results.append(parsed_dict)
        except json.JSONDecodeError:
            pass

    return results


def route_templates(
    route_selection_outputs: list[str],
    prompts: list[str],
    restrictive_system_instructions: str,
    env: Environment,
    no_violation_template_name: str,
    potential_violation_template_name_first_turn: str,
    potential_violation_template_name_second_turn: str,
    direct_violation_template_name: str,
    fix_route: str = "",
    fix_tip: str = "",
    fix_check_result: str = "",
):

    system_check_results = []
    routes = []
    system_tips = []

    possible_route_values = [
        "no_to_minimal_risk",
        "potential_violation",
        "direct_violation",
    ]
    if fix_route:
        assert (
            fix_route in possible_route_values
        ), f"fix_route must be one of {possible_route_values}"
    # Process each violation answer
    for ix, violation in enumerate(route_selection_outputs):
        try:
            violation_lst_dict = extract_and_eval_json(violation)
            if len(violation_lst_dict) == 0:
                raise ValueError(
                    f"Could not parse the violation as a dict.\
                    The problematic violation string is:\n {violation} \
                    "
                )
            else:
                violation_dict = violation_lst_dict[0]
            # transform the keys to lower case
            violation_dict = {k.lower(): v for k, v in violation_dict.items()}
            system_check_result = (
                violation_dict["system_check_result"]
                if not fix_check_result
                else fix_check_result
            )

            route = (
                violation_dict["route"].lower().replace(" ", "")
                if not fix_route
                else fix_route
            )
            if route not in possible_route_values:
                print(
                    f"Route {route} not in possible values. Defaulting to 'potential_violation'"
                )
                route = "potential_violation"
            system_tip = violation_dict["system_tip"] if not fix_tip else fix_tip
            system_check_results.append(system_check_result)
            routes.append(route)
            system_tips.append(system_tip)
        except:
            # Handle parsing errors and set default values
            print(
                f"Could not parse\n: {violation}\n at index {ix} as a dict \
                Defaulting to 'potential_violation' for route, str(violation) for system_check_result and 'Not avaliable"
            )
            routes.append("potential_violation")
            system_check_results.append(str(violation))
            system_tip = "Not avaliable" if not fix_tip else fix_tip
            system_tips.append(system_tip)

    # Load the templates for each possible route
    no_risk_template = env.get_template(
        no_violation_template_name
    )  # input: user_input = prompts[i]
    potential_violation_templates = {
        "analysis": env.get_template(
            potential_violation_template_name_first_turn
        ),  # inputs to render: system_instructions=restrictive_system_instructions,  user_input = prompts[i]
        "answer": env.get_template(
            potential_violation_template_name_second_turn
        ),  # inputs to render: system_tip[i]
    }
    refusal_template = env.get_template(
        direct_violation_template_name
    )  # nothing to render

    output_conv = []
    # Generate the conversation output based on the route
    # for no_risk and refusal the inner list has length 1
    # for potential_violation the inner list has length 3
    for ix, route in enumerate(routes):
        if route == "no_to_minimal_risk":
            output_conv.append(
                [
                    no_risk_template.render(
                        user_input=prompts[ix], system_tip=system_tips[ix]
                    )
                ]
            )
        elif route == "potential_violation":
            output_conv.append(
                [
                    potential_violation_templates["analysis"].render(
                        system_instructions=restrictive_system_instructions,
                        user_input=prompts[ix],
                    ),  # user
                    # system_check_results[ix],  # assistant
                    f"<SYSTEM_TIP>\n{system_tips[ix]}\n</SYSTEM_TIP>\n<SYSTEM_CHECK_RESULT>\n{system_check_results[ix]}\n</SYSTEM_CHECK_RESULT> ",  # assistant
                    potential_violation_templates["answer"].render(
                        system_tip=system_tips[ix]
                    ),  # user
                ]
            )
        elif route == "direct_violation":
            output_conv.append(
                [
                    refusal_template.render(
                        user_input=prompts[ix], system_tip=system_tips[ix]
                    )
                ]
            )
    return output_conv, routes, system_check_results, system_tips
