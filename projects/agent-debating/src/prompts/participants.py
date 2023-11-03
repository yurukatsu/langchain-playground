CREATE_PARTICIPANT_PROMPT_TEMPLATE = """Conversation situations are given as follows:
{situation}

You are to create {n_participants} hypothetical participants (personas) under the above situation.
Try to diversify the attributes of each participant as much as possible.
Do not add anything else.

{format_instructions}"""
