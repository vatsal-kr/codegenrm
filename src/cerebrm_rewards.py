import ast
import re
from typing import List

import numpy as np

def extract_boxed_contents_score10(text: str) -> List[int]:
    """
    Extracts all contents within \\boxed{...} from a given text string,
    after normalizing braces.
    """
    # Match \boxed{...} with non-greedy content
    pattern = r"\\boxed\{(.*?)\}"
    matches = re.search(pattern, text)
    try:
        matches = ast.literal_eval(matches.group(1))
        assert isinstance(matches, list) and all(isinstance(x, int) for x in matches)
    except Exception:
        matches = []
    return matches


def extract_boxed_contents_list(text: str) -> str:
    """
    Extracts all contents within \\boxed{...} from a given text string,
    after normalizing braces.
    """
    # Match \boxed{...} with non-greedy content
    pattern = r"\\boxed\{(.*?)\}"
    matches = re.findall(pattern, text)

    if len(matches) == 1:
        return matches[0]
    return None



def soft_overlong_punishment(completion_ids, L_max, L_cache, **kwargs):
    # taken from https://github.com/huggingface/trl/issues/3130
    rewards = []
    for ids in completion_ids:
        completion_length = len(ids)
        if completion_length <= L_max - L_cache:
            rewards.append(0.0)
        elif L_max - L_cache < completion_length <= L_max:
            rewards.append((L_max - L_cache - completion_length) / L_cache)
        else:
            rewards.append(-1.0)
    return rewards


def list_format_reward(completions, num_candidates, **kwargs):
    """
    Assigns a reward based on whether the model's output is correctly formatted.
    0 if correctly formatted, else -1
    Args:
        completions (List[List[Dict[str,str]]]): A list of model completions, each being a list of dictionaries with keys "role" and "content". In practice, each completion list has only one dictionary.
    Returns:
        List[float]: A list of rewards for each completion.
    """
    pattern_dict = {
        2: r"^<think>(?!.*<think>)(.*?)</think>\s*\\boxed{[AB]}$",
        3: r"^<think>(?!.*<think>)(.*?)</think>\s*\\boxed{[ABC]}$",
        4: r"^<think>(?!.*<think>)(.*?)</think>\s*\\boxed{[ABCD]}$",
        5: r"^<think>(?!.*<think>)(.*?)</think>\s*\\boxed{[ABCDE]}$",
    }

    completion_contents = [completion[0]["content"].strip() for completion in completions]
    matches = [re.match(pattern_dict[nc], content, re.DOTALL) for content, nc in zip(completion_contents, num_candidates)]
    return [0.0 if match else -1.0 for match in matches]


def list_reward(completions, chosen_answer, **kwargs):
    """
    Calculates the reward for each completion based on the chosen answer.
    +1 if the chosen answer matches the ground truth, else 0.

    Args:
        completions (List[List[Dict[str,str]]]): A list of model completions, each being a list of dictionaries with keys "role" and "content". In practice, each completion list has only one dictionary.
        chosen_answer (List[str]): A list of ground truth answers.

    Returns:
        List[float]: A list of rewards for each completion.
    """
    contents = [completion[0]["content"].split("</think>")[-1].strip() for completion in completions]
    contents = [extract_boxed_contents_list(completion) for completion in contents]
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, chosen_answer)]


def list_format_reward_cot(completions, num_candidates, **kwargs):
    """
    Assigns a reward based on whether the model's output is correctly formatted.
    0 if correctly formatted, else -1
    Args:
        completions (List[List[Dict[str,str]]]): A list of model completions, each being a list of dictionaries with keys "role" and "content". In practice, each completion list has only one dictionary.
    Returns:
        List[float]: A list of rewards for each completion.
    """
    pattern_dict = {
        2: r"\\boxed{[AB]}$",
        3: r"\\boxed{[ABC]}$",
        4: r"\\boxed{[ABCD]}$",
        5: r"\\boxed{[ABCDE]}$",
    }
    completion_contents = [completion[0]["content"].strip() for completion in completions]
    matches = [re.search(pattern_dict[nc], content, re.DOTALL) for content, nc in zip(completion_contents, num_candidates)]
    return [0.0 if match else -1.0 for match in matches]


def list_reward_cot(completions, chosen_answer, **kwargs):
    """
    Calculates the reward for each completion based on the chosen answer.
    +1 if the chosen answer matches the ground truth, else 0.

    Args:
        completions (List[List[Dict[str,str]]]): A list of model completions, each being a list of dictionaries with keys "role" and "content". In practice, each completion list has only one dictionary.
        chosen_answer (List[str]): A list of ground truth answers.

    Returns:
        List[float]: A list of rewards for each completion.
    """
    contents = [completion[0]["content"].strip() for completion in completions]
    contents = [extract_boxed_contents_list(completion) for completion in contents]
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, chosen_answer)]


def list_reward_with_distance(completions, pass_rates, num_candidates, **kwargs):
    """
    Assigns a +1 reward if the model's chosen answer matches the ground truth.
    For mismatches, assigns a negative reward based on the distance from the correct answer.
    The further the chosen answer is from the correct one, the more negative the reward.
    The worst possible reward is (N-1)/N = -0.8 (we know that the longest list has N = 5 candidates).
    For list lengths x in [2, 3, 4, 5], the rewards are as follows:
        If the model chooses the x-th best answer (ie the worst answer), it gets a reward of -1*(N-1)/N
        If the model chooses the n-th best answer, where 1<n<x it gets a reward of -1*(N-n)/N
        If the model chooses the correct answer, it gets a reward of +1
        If the model chooses an answer outside the provided options, it gets a reward of -1

    Args:
        completions (List[List[Dict[str,str]]]): A list of model completions, each being a list of dictionaries with keys "role" and "content". In practice, each completion list has only one dictionary.
        pass_rates (List[List[float]]): A list of pass rates for each candidate.
        num_candidates (List[int]): The number of candidates each completion had to consider.

    Returns:
        List[float]: A list of rewards for each completion.
    """
    rewards = []
    contents = [c[0]["content"].split("</think>")[-1].strip() for c in completions]
    contents = [extract_boxed_contents_list(completion) for completion in contents]
    for completion, pass_rate, num_candidate in zip(contents, pass_rates, num_candidates):
        # Generate possible answers up to num_candidates
        possible_answers = ["A", "B", "C", "D", "E"][:num_candidate]
        possible_rewards = [1, -0.8, -0.6, -0.4, -0.2][:num_candidate]
        possible_rewards = sorted(possible_rewards, reverse=True)
        # Map answers to their pass rates
        answer_to_pr = {x: y for x, y in zip(possible_answers, pass_rate)}
        # Sort by pass rate (descending)
        answer_to_pr = dict(sorted(answer_to_pr.items(), key=lambda item: item[1], reverse=True))
        # Use this ordered dictionary to assign rewards - highest (chosen) gets 1 and worst gets -0.8
        answer_to_reward = {x: y for x, y in zip(answer_to_pr.keys(), possible_rewards)}

        rewards.append(answer_to_reward.get(completion, -1))

    return rewards


def grm_correctness_reward(completions, chosen_position, num_candidates, **kwargs):
    """
    Assigns a +1 reward if the model's chosen answer matches the ground truth, and an additional +1 if the correct one is assigned a score of 10.
    Model completions consist of a "think" section followed by a boxed list of scores for each candidate.
    The index of the highest score in the boxed list is taken as the model's chosen answer.
    If the chosen answer matches the ground truth position, and the highest score is 10, a reward of +2 is given
    If the chosen answer matches the ground truth position, but the highest score is less than 10, a reward of +1 is given
    If the chosen answer does not match the ground truth position, a reward of 0 is
    If there is no unique maximum, or a mismatch in the number of scores and candidates, a reward of 0 is given
    Args:
        completions (List[List[Dict[str,str]]]): A list of model completions, each being a list of dictionaries with keys "role" and "content". In practice, each completion list has only one dictionary.
        chosen_position (List[int]): A list of ground truth positions.

    Returns:
        List[float]: A list of rewards for each completion.
    """
    contents = [completion[0]["content"].split("</think>")[-1].strip() for completion in completions]
    # contents is something like [\boxed{[8,4,3]}, \boxed{[3,7,1,1]},...]
    contents = [extract_boxed_contents_score10(completion) for completion in contents]
    rewards = []
    for completion, num_candidate, gt_position in zip(contents, num_candidates, chosen_position):
        if len(completion) != num_candidate:
            rewards.append(-1.0)
            continue
        # model verdict is considered only if there is a unique max
        model_verdict = completion.index(max(completion)) if completion and completion.count(max(completion)) == 1 else -1
        if model_verdict == gt_position and max(completion) == 10:
            rewards.append(2.0)
        elif model_verdict == gt_position:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards


def judgelrm_content_reward(completions, pass_rates, **kwargs):
    """
    Assigns a reward based on the content of the model completions and the ground truth pass rates.
    Only valid for 2 candidates.
    Reward consists of 3 components:
        R_relative: +2 if the relative ordering of scores matches the ground truth, -1.5 if not
        R_absolute: +1 if absolute scores match exactly, +0.6 if the total difference is within 2 and relative ordering is correct, else 0
        R_confidence: +0.2 if the model's confidence (difference between scores) is greater than or equal to the ground truth confidence and relative ordering is correct, else 0
    Args:
        completions (List[List[Dict[str,str]]]): A list of model completions, each being a list of dictionaries with keys "role" and "content". In practice, each completion list has only one dictionary.
        pass_rates (List[List[float]]): A list of ground truth pass rates, each being a list of floats.

    Returns:
        List[float]: A list of rewards for each completion.
    """

    contents = [completion[0]["content"].split("</think>")[-1].strip() for completion in completions]
    gt_scores = [[round(x * 10) for x in pr] for pr in pass_rates]  # convert to 10-point scale
    # contents is something like [\boxed{[8,4]}, \boxed{[3,7]},...]
    contents = [extract_boxed_contents_score10(completion) for completion in contents]
    # contents is something like [[8,4], [3,7]]
    content_rewards = []
    for content, gt_score in zip(contents, gt_scores):
        if not len(content) == 2:
            content_rewards.append(-2.0)
            continue
        score_a = content[0]
        score_b = content[1]
        gt_score_a = gt_score[0]
        gt_score_b = gt_score[1]

        if np.sign(score_a - score_b) == np.sign(gt_score_a - gt_score_b):
            r_rel = 2.0
        else:
            r_rel = -1.5
        if abs(score_a - gt_score_a) + abs(score_b - gt_score_b) == 0:
            r_abs = 1.0
        elif r_rel == 2 and abs(score_a - gt_score_a) + abs(score_b - gt_score_b) <= 2:
            r_abs = 0.6
        else:
            r_abs = 0.0

        if r_rel == 2 and abs(score_a - score_b) >= abs(gt_score_a - gt_score_b):
            r_conf = 0.2
        else:
            r_conf = 0.0
        content_rewards.append(round(r_rel + r_abs + r_conf, 2))
    return content_rewards


def judgelrm_format_reward(completions, **kwargs):
    """
    Assigns a reward based on whether the model's output is correctly formatted.
    +1 if correctly formatted, -0.5 if correctly formatted but two invalid scores, else -1
    Args:
        completions (List[List[Dict[str,str]]]): A list of model completions, each being a list of dictionaries with keys "role" and "content". In practice, each completion list has only one dictionary.
    Returns:
        List[float]: A list of rewards for each completion.
    """
    pattern = r"^<think>(?!.*<think>)(.*?)</think>\s*\\boxed{\[\d+,\s*\d+\]}$"
    completion_contents = [completion[0]["content"].strip() for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    verdicts = [extract_boxed_contents_score10(x.split("</think>")[-1].strip()) for x in completion_contents]
    format_rewards = []
    for match, verdict in zip(matches, verdicts):
        if match and len(verdict) == 2 and all(0 <= x <= 10 for x in verdict):
            format_rewards.append(1.0)
        elif match and len(verdict) == 2 and not all(0 <= x <= 10 for x in verdict):
            format_rewards.append(-0.5)
        else:
            format_rewards.append(-1.0)
    return format_rewards


def grm_format_reward(completions, num_candidates, **kwargs):
    """
    Assigns a reward based on whether the model's output is correctly formatted.
    0 if correctly formatted, else -1
    Args:
        completions (List[List[Dict[str,str]]]): A list of model completions, each being a list of dictionaries with keys "role" and "content". In practice, each completion list has only one dictionary.
    Returns:
        List[float]: A list of rewards for each completion.
    """
    pattern_dict = {
        2: r"^<think>(?!.*<think>)(.*?)</think>\s*\\boxed{\[\d+,\s*\d+\]}$",
        3: r"^<think>(?!.*<think>)(.*?)</think>\s*\\boxed{\[\d+,\s*\d+,\s*\d+\]}$",
        4: r"^<think>(?!.*<think>)(.*?)</think>\s*\\boxed{\[\d+,\s*\d+,\s*\d+,\s*\d+\]}$",
        5: r"^<think>(?!.*<think>)(.*?)</think>\s*\\boxed{\[\d+,\s*\d+,\s*\d+,\s*\d+,\s*\d+\]}$",
    }

    completion_contents = [completion[0]["content"].strip() for completion in completions]
    matches = [re.match(pattern_dict[nc], content, re.DOTALL) for content, nc in zip(completion_contents, num_candidates)]
    return [0.0 if match else -1.0 for match in matches]
