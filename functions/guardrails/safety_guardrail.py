from better_profanity import profanity

def check_for_profanity(user_input):
    """Checks if the user input contains profanity and returns the profane words."""
    # Using profanity.contains_profanity() to detect profanity in the user input
    profane_words = [word for word in user_input.split() if profanity.contains_profanity(word)]

    if profane_words:
        return True, profane_words  # Return True and the list of bad words
    return False, []

def safety_guardrail(user_input):
     """Checks if the user input contains harmful language."""
     is_profane, profane_words = check_for_profanity(user_input)
     if is_profane:
        return True, profane_words  # Return the violation
     return False, []