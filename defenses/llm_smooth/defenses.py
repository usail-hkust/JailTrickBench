import torch
import copy
import random
import numpy as np
import math
import llm_smooth.perturbations as perturbations
from utils.test_utils import test_prefixes
class Defense:

    

    def __init__(self, target_model):
        self.target_model = target_model
        self.TEST_PREFIXES =  test_prefixes

    def is_jailbroken(self, s):
        """Returns True if a prompt results in a jailbreak; False otherwise."""

        return not any([
            prefix.lower() in s.lower() for prefix in self.TEST_PREFIXES
        ])

class SmoothLLM(Defense):

    """SmoothLLM defense.
    
    Title: SmoothLLM: Defending Large Language Models Against 
                Jailbreaking Attacks
    Authors: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas
    Paper: https://arxiv.org/abs/2310.03684
    """

    def __init__(self, 
        target_model,
        pert_type,
        pert_pct,
        num_copies
    ):
        super(SmoothLLM, self).__init__(target_model)
        
        self.num_copies = num_copies
        self.perturbation_fn = vars(perturbations)[pert_type](
            q=pert_pct
        )

    @torch.no_grad()
    def __call__(self, prompt, batch_size=5, max_new_len=100):

        all_inputs = []
        for _ in range(self.num_copies):
            prompt_copy = copy.deepcopy(prompt)
            #prompt_copy.perturb(self.perturbation_fn)
            prompt_copy = self.perturbation_fn(prompt_copy)
            all_inputs.append(prompt_copy)

        # Iterate each batch of inputs
        all_outputs = []
        for i in range(math.ceil(self.num_copies / batch_size)):

            # Get the current batch of inputs
            batch = all_inputs[i * batch_size:(i+1) * batch_size]

            # Run a forward pass through the LLM for each perturbed copy
            batch_outputs = self.target_model(
                batch=batch, 
            )

            all_outputs.extend(batch_outputs)
            torch.cuda.empty_cache()

        # Check whether the outputs jailbreak the LLM
        are_copies_jailbroken = [self.is_jailbroken(s) for s in all_outputs]
        if len(are_copies_jailbroken) == 0:
            raise ValueError("LLM did not generate any outputs.")

        outputs_and_jbs = zip(all_outputs, are_copies_jailbroken)

        # Determine whether SmoothLLM was jailbroken
        jb_percentage = np.mean(are_copies_jailbroken)
        smoothLLM_jb = True if jb_percentage > 0.5 else False

        # Pick a response that is consistent with the majority vote
        majority_outputs = [
            output for (output, jb) in outputs_and_jbs 
            if jb == smoothLLM_jb
        ]
        return random.choice(majority_outputs)



