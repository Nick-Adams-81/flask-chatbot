import mlflow
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.chat_bot import chat_bot

# Test dataset (Expected vs Actual Responses)
benchmark_tests = [
    {
        "question": "What is the minimum bet rule in a tournament?",
        "expected": "In a tournament, the minimum bet or raise must be at least the size of the previous bet or raise. For example, if a player raises 300, the next raise must also be at least 300. This rule ensures that each successive bet or raise matches or exceeds the largest in the current betting round."
    },
    {
        "question": "Can I talk about my hand while in play?",
        "expected": "No, you may not discuss your hand while it is in play. Players must protect other players in the tournament and are prohibited from discussing the contents of live or mucked hands. The 'one-player-to-a-hand' rule also prohibits showing your hand or discussing strategy with anyone else."
    },
    {
        "question": "What happens if a player acts out of turn?",
        "expected": "If a player acts out of turn, the action will be backed up to the correct player in order."
    },
    {
        "question": "What happens if a player goes all-in during a tournament?",
        "expected": "When a player goes all-in, they commit all their chips to the pot. The rest of the players can continue betting until the betting round ends. A player who goes all-in is eligible to win the amount of chips in the pot corresponding to their contribution. Any side pots will be contested among the remaining players."
    },
    {
        "question": "What happens if there is a misdeal in a tournament?",
        "expected": "In the event of a misdeal, the hand is considered void and a new hand is dealt. All players receive their original cards back and the betting round starts fresh. A misdeal results in the hand being canceled and all players' cards returned, after which the dealer redeals the cards."
    },
    {
        "question": "What happens if there is a discrepancy in a player's chip count during a tournament?",
        "expected": "If there is a chip count discrepancy, the dealer will verify the correct number of chips. If an error is found, the chips are adjusted to reflect the correct count. In case of a chip count error, the dealer will attempt to reconcile the discrepancy by reviewing previous bets and counts."
    },
]

# Set up MLflow experiment
mlflow.set_experiment("TDA Chatbot Benchmark tests")

# Start a top-level run for benchmarking
with mlflow.start_run(run_name=f"benchmark_run"):
    total_tests = len(benchmark_tests)
    correct = 0

    for idx, test in enumerate(benchmark_tests, 1):
        question = test["question"]
        expected_responses = test["expected"]
        
        # Call the chatbot with the actual logic
        document_path = "/Users/nicholasadams/Code/flask-chatbot/data/tournament-rules.txt"  # Specify the path to the document for the chatbot
        actual_response = chat_bot(document_path, question)

        # Check if the response is similar to any expected response
        is_correct = any(expected.lower() in actual_response.lower() for expected in expected_responses)

        # Log unique parameters and metrics for each test case
        mlflow.log_param(f"Test_{idx}_Question", question)  # Log the question
        mlflow.log_param(f"Test_{idx}_Expected_Response", ", ".join(expected_responses))  # Log expected responses
        mlflow.log_param(f"Test_{idx}_Actual_Response", actual_response)  # Log actual response
        mlflow.log_metric(f"Test_{idx}_Correct", int(is_correct))  # Log correctness for each question

        if is_correct:
            correct += 1

    # Calculate overall accuracy and log it in the top-level run
    accuracy = correct / total_tests
    mlflow.log_metric("Benchmark_Accuracy", accuracy)

    # Print the final result
    print(f"Benchmarking completed. Accuracy: {accuracy:.2%}")