import gradio as gr
from jinja2 import Environment, FileSystemLoader
import json
from model import LitellmModel
from routing import route

# Load cached results
with open("cached_results.json", "r") as f:
    cached_results = json.load(f)

# Define constants
model = LitellmModel(model_id="mistral/open-mistral-7b")
env = Environment(loader=FileSystemLoader("templates"))

def process_input(prompt, cached_prompt):
    """
    Process the input prompt, giving priority to cached prompts if selected.

    Args:
        prompt (str): The user-entered prompt from the text input.
        cached_prompt (str): The selected cached prompt from the dropdown.

    Returns:
        tuple: Contains the processed results and UI update information.
    """
    # Use the cached prompt if one is selected, otherwise use the user-entered prompt
    if cached_prompt:
        prompt = cached_prompt

    # Call the route function to process the prompt
    result = route(prompt, model, env, cached_results)

    # Prepare button updates based on the routing result
    button_updates = [
        gr.update(variant="secondary"),
        gr.update(variant="secondary"),
        gr.update(variant="secondary"),
    ]
    if result["route"] == "no_to_minimal_risk":
        button_updates[0] = gr.update(variant="primary")
    elif result["route"] == "potential_violation":
        button_updates[1] = gr.update(variant="primary")
    elif result["route"] == "direct_violation":
        button_updates[2] = gr.update(variant="primary")

    # Return all the results and UI updates
    return (
        result["vanilla_result"],
        result["primeguard_result"],
        *button_updates,
        result["system_check"],
        result["system_tip"],
        result["reevaluation"],
        prompt,  # Return the prompt to update the input field
    )


css = """
.route-button { height: 50px; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("# 🤺 PrimeGuard Demo 🤺")

    with gr.Row():
        with gr.Column(scale=3):
            prompt_input = gr.Textbox(
                label="Enter your prompt", lines=3, placeholder="You can't break me"
            )
            submit_btn = gr.Button("Submit", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Cached Examples")
            cached_prompts = gr.Dropdown(
                choices=[item["prompt"] for item in cached_results],
                label="Select a cached prompt",
                allow_custom_value=True,
            )

    with gr.Row():
        vanilla_output = gr.Textbox(
            label="Mistral 7B Defended with System Prompt 🤡",
            lines=5,
            interactive=False,
        )
        primeguard_output = gr.Textbox(
            label="Mistral 7B Defended with 🤺 PrimeGuard 🤺",
            lines=5,
            interactive=False,
        )

    gr.Markdown("## PrimeGuard Details")

    with gr.Row():
        no_risk = gr.Button(
            "No to Minimal Risk", variant="secondary", elem_classes=["route-button"]
        )
        potential_violation = gr.Button(
            "Potential Violation", variant="secondary", elem_classes=["route-button"]
        )
        direct_violation = gr.Button(
            "Direct Violation", variant="secondary", elem_classes=["route-button"]
        )

    with gr.Column():
        system_check = gr.Textbox(
            label="System Check Result", lines=3, interactive=False
        )
        system_tip = gr.Textbox(label="System Tip", lines=3, interactive=False)
        reevaluation = gr.Textbox(label="Reevaluation", lines=3, interactive=False)

    with gr.Row():
        gr.HTML(
            """<a href="https://www.dynamofl.com" target="_blank">
                    <p align="center">
                        <img src="https://bookface-images.s3.amazonaws.com/logos/4decc4e1a1e133a40d326cb8339c3a52fcbfc4dc.png" alt="Dynamo" width="200">
                    </p>
                </a>
        """,
            elem_id="ctr",
        )

    def update_ui(
        vanilla_result,
        primeguard_result,
        no_risk_update,
        potential_violation_update,
        direct_violation_update,
        system_check_result,
        system_tip,
        reevaluation,
        prompt,
    ):
        return [
            vanilla_result,
            primeguard_result,
            no_risk_update,
            potential_violation_update,
            direct_violation_update,
            system_check_result,
            system_tip,
            reevaluation,
            prompt,  # Update the prompt input field
        ]

    def reset_cached_prompt():
        return gr.update(value=None)

    submit_btn.click(
        fn=process_input,
        inputs=[prompt_input, cached_prompts],
        outputs=[
            vanilla_output,
            primeguard_output,
            no_risk,
            potential_violation,
            direct_violation,
            system_check,
            system_tip,
            reevaluation,
            prompt_input,
        ],
    ).then(
        fn=update_ui,
        inputs=[
            vanilla_output,
            primeguard_output,
            no_risk,
            potential_violation,
            direct_violation,
            system_check,
            system_tip,
            reevaluation,
            prompt_input,
        ],
        outputs=[
            vanilla_output,
            primeguard_output,
            no_risk,
            potential_violation,
            direct_violation,
            system_check,
            system_tip,
            reevaluation,
            prompt_input,
        ],
    ).then(
        fn=reset_cached_prompt,
        inputs=[],
        outputs=[cached_prompts],
    )

    # Add an event listener for the cached_prompts dropdown
    cached_prompts.change(
        fn=process_input,
        inputs=[prompt_input, cached_prompts],
        outputs=[
            vanilla_output,
            primeguard_output,
            no_risk,
            potential_violation,
            direct_violation,
            system_check,
            system_tip,
            reevaluation,
            prompt_input,
        ],
    ).then(
        fn=update_ui,
        inputs=[
            vanilla_output,
            primeguard_output,
            no_risk,
            potential_violation,
            direct_violation,
            system_check,
            system_tip,
            reevaluation,
            prompt_input,
        ],
        outputs=[
            vanilla_output,
            primeguard_output,
            no_risk,
            potential_violation,
            direct_violation,
            system_check,
            system_tip,
            reevaluation,
            prompt_input,
        ],
    )

demo.queue(max_size=20)
demo.launch(max_threads=40)
