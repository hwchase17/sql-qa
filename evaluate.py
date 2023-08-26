from langchain.smith import RunEvalConfig, run_on_dataset
from langsmith import Client

from utils import get_agent

if __name__ == "__main__":

    client = Client()
    eval_config = RunEvalConfig(
        evaluators=[
            "qa"
        ],
    )
    chain_results = run_on_dataset(
        client,
        dataset_name="misspelled-examples",
        concurrency_level=1,
        llm_or_chain_factory=get_agent,
        evaluation=eval_config,
    )