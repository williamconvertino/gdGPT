# import torch
# import os
# import time

# from src.util import get_time_remaining

# EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/evaluations')

# MODEL = 'gpt-4o'

# SYSTEM_PROMPT = "You are an expert writing evaluator designed to assess student story completions. Your role is to provide constructive, fair, and detailed evaluations based on specific rubric criteria."

# USER_PROMPT = """
# In the following exercise, the student is given a pre-written beginning of a story. The student needs to complete this story. The exercise tests the student´s language abilities and creativity.

# Here is the pre-written beginning:

# <PROVIDED BEGINNING>
# [STORY_BEGIN]
# </PROVIDED BEGINNING>

# And here is the students response:

# <STUDENT RESPONSE>
# [STORY_END]
# </STUDENT RESPONSE>

# First, provide a concise qualitative assessment about the student's writing. Then, give the writing a grade out of 10. These assessments should be done for each of the following rubric items:

# 1. Grammar:
# * Is the writing grammatically correct?
# * Evaluate syntax, punctuation, and sentence structure.
# 2. Consistency:
# * Is the student's writing consistent with the provided beginning of the story?
# * How well does the student complete the final sentence of the prescribed beginning?
# 3. Plot:
# * Does the plot of the student's writing make sense (regardless of the provided beginning)?
# 4. Creativity: 
# * How creative is the student's writing?

# Format your response as follows:

# <GRAMMAR>
# [Qualitative assessment of grammar]
# </GRAMMAR>
# <GRAMMAR_GRADE>
# [Grade out of 10]
# </GRAMMAR_GRADE>

# <CONSISTENCY>
# [Qualitative assessment of consistency]
# </CONSISTENCY>
# <CONSISTENCY_GRADE>
# [Grade out of 10]
# </CONSISTENCY_GRADE>

# <PLOT>
# [Qualitative assessment of plot]
# </PLOT>
# <PLOT_GRADE>
# [Grade out of 10]
# </PLOT_GRADE>

# <CREATIVITY>
# [Qualitative assessment of creativity]
# </CREATIVITY>
# <CREATIVITY_GRADE>
# [Grade out of 10]
# </CREATIVITY_GRADE>

# Provide your assessment below:
# """

# def generate_gpt4o_inputs(model, tokenizer, test_dataset, num_generations=10):
  
#   i = 0
#   num_skipped = 0
#   start_time = time.time()
  
#   eval_items = []
  
#   for batch in enumerate(test_dataset):
    
#     if i >= num_generations:
#       break
    
#     sequence = batch[0]

#     input_size = model.config.context_size // 2

#     model_input = sequence[:input_size]
    
#     if tokenizer.eos_token_id in model_input: # Exclude sequences with incomplete prompts, to avoid confusion in the GPT-4o evaluation
#       num_skipped += 1
#       continue
    
#     with torch.no_grad():
#       model.eval()
      
#       story_begin = tokenizer.decode(model_input.tolist())
#       story_true_end = tokenizer.decode(sequence[input_size:].tolist())
      
#       beam_search_sequence = model.beam_search(model_input.unsqueeze(0))
#       beam_search_sequence = beam_search_sequence[input_size:].tolist()
#       eos_token_id = beam_search_sequence.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in beam_search_sequence else len(beam_search_sequence)
#       beam_search_sequence = beam_search_sequence[:eos_token_id] # Remove EOS token if present
#       if len(beam_search_sequence) < 2: # Exclude sequences with less than 2 tokens, to avoid confusion in the GPT-4o evaluation
#         num_skipped += 1
#         continue

#       story_beam_end = tokenizer.decode(beam_search_sequence)
      
#       true_prompt = USER_PROMPT.replace('[STORY_BEGIN]', story_begin).replace('[STORY_END]', story_true_end)
#       beam_prompt = USER_PROMPT.replace('[STORY_BEGIN]', story_begin).replace('[STORY_END]', story_beam_end)
      
#       eval_items.append({
#         "custom_id": f"request_{i}_true",
#         "method": "POST",
#         "url": "/v1/chat/completions",
#         "body": {
#           "model": MODEL,
#           "messages": [
#             {
#               "role": "system", "content": SYSTEM_PROMPT
#             },
#             {
#               "role": "user", "content": true_prompt
#             }
#           ],
#           "max_tokens": 1000
#         }
#       })
      
#       eval_items.append({
#         "custom_id": f"request_{i}_beam",
#         "method": "POST",
#         "url": "/v1/chat/completions",
#         "body": {
#           "model": MODEL,
#           "messages": [
#             {
#               "role": "system", "content": SYSTEM_PROMPT
#             },
#             {
#               "role": "user", "content": beam_prompt
#             }
#           ],
#           "max_tokens": 1000
#         }
#       })
      
#       i += 1
#       time_remaining = get_time_remaining(start_time, i, num_generations)
#       print(f"\r{i}/{num_generations} ({100 * i / num_generations:.2f}%) | Time Remaining: {time_remaining}", end='')
      
#   with open(f'{EVAL_DIR}/gpt4o_eval_input.jsonl', 'w') as f:
#     for item in eval_items:
#       f.write(f"{item}\n")
      
#   print(f"Generated {i} GPT-4o evaluation inputs, skipped {num_skipped} sequences.")