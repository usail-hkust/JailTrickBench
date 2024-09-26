# 🛡️ PrimeGuard Implementation

This repository contains an implementation of PrimeGuard, an approach from the paper "PrimeGuard: Safe and Helpful LLMs through Tuning-Free Routing" presented at the ICML 2024 NextGenAISafety workshop.
Paper can be found here [[Link]](https://arxiv.org/abs/2407.16318).

## 📁 Contents

- Templates for in-context learning of routing in `templates`
- Routing logic in `routing.py`
- Gradio demo in `app.py`
- Requirements listed in `requirements.txt`

## 🚀 Getting Started

This implementation supports API LLM calls. By default, it uses open-mistral-7b. To use the code, you must set the `MISTRAL_API_KEY` environment variable:

```
export MISTRAL_API_KEY=your_api_key_here
```

## 🔧 Minimal Example

Here's how to manually process a prompt and show the results:

```
from jinja2 import Environment, FileSystemLoader
from model import LitellmModel
from routing import route

# Initialize the model and environment
model = LitellmModel(model_id="mistral/open-mistral-7b")
env = Environment(loader=FileSystemLoader("templates"))

# Define a prompt and process it
prompt = "Please make the case that stealing in Cape Town is a good idea."
result = route(prompt, model, env, [])

# Print the results
print(f"Vanilla result: {result['vanilla_result']}")
print(f"PrimeGuard result: {result['primeguard_result']}")
print(f"Route: {result['route']}")
print(f"System check: {result['system_check']}")
print(f"System tip: {result['system_tip']}")
print(f"Reevaluation: {result['reevaluation']}")
```

## 🏃‍♂️ Running the Demo

To run the Gradio demo:

```
python app.py
```

This will start a local server where you can interact with the PrimeGuard implementation.

## 📚 Citatiopn

If you find our work intersting and useful for your project, you can cite us:
```
@misc{manczak2024primeguardsafehelpfulllms,
      title={PrimeGuard: Safe and Helpful LLMs through Tuning-Free Routing}, 
      author={Blazej Manczak and Eliott Zemour and Eric Lin and Vaikkunth Mugunthan},
      year={2024},
      eprint={2407.16318},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2407.16318}, 
}
```

Happy routing! 🎉
