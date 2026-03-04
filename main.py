from src.preprocessing import preprocess
from src.monthly_pipeline import MonthlyPipeline
from src.benchmark import benchmark

if __name__ == "__main__":

    bank, check = preprocess(
        "data/bank_statements.csv",
        "data/check_register.csv"
    )

    pipeline = MonthlyPipeline(bank, check)

    benchmark(pipeline.run)