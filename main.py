import argparse
import logging
import os

from dotenv import load_dotenv

from src.config import Config
from src.logger import get_logger, set_log_level
from src.pipeline import RAGPipeline

load_dotenv()


def run_demo(pipeline: RAGPipeline):
    logger = get_logger()

    logger.info("Starting Multi-Agent RAG Demonstration")
    print("\n" + "=" * 60)
    print("🚀 MULTI-AGENT RAG DEMONSTRATION")
    print("=" * 60)

    demo_queries = pipeline.questions
    if not demo_queries:
        print("No questions found in loaded dataset.")
        return

    # Use only the first query for demo
    query = demo_queries[0]
    print(f"\n{'─' * 60}")
    result = pipeline.query(query)
    print(result.to_readable())

    print("\n" + "=" * 60)
    logger.info("Demo complete")
    print("✅ Demo complete!")


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent RAG System")
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key",
    )
    parser.add_argument("--query", "-q", type=str, help="Single query to process")
    parser.add_argument(
        "--demo", action="store_true", help="Run demo with dataset queries"
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=10,
        help="Number of RAMDocs samples to load (default: 10)",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation against RAMDocs ground truth",
    )
    args = parser.parse_args()

    if not args.api_key:
        logger = get_logger()
        logger.error("OpenAI API key required. Use --api-key or set OPENAI_API_KEY")
        print("  ERROR: OpenAI API key required.")
        print("   Use --api-key YOUR_KEY or set OPENAI_API_KEY environment variable")
        return 1

    config = Config.from_env(api_key=args.api_key)
    config.verbose = not args.quiet

    if args.quiet:
        set_log_level(logging.WARNING)

    if args.evaluate:
        from src.evaluate import run_evaluation

        metrics = run_evaluation(api_key=args.api_key, num_samples=args.samples)
        print(
            f"\n📊 RESULTS: Accuracy={metrics['accuracy']:.1%}, F1={metrics['f1']:.2f}, Misinfo={metrics['misinfo_rate']:.1%}"
        )
        return 0

    logger = get_logger()
    logger.info("Initializing Multi-Agent RAG Pipeline...")
    pipeline = RAGPipeline(config)
    pipeline.load_and_index(num_samples=args.samples)

    if args.demo:
        run_demo(pipeline)
    elif args.query:
        result = pipeline.query(args.query)
        print(result.to_readable())
    else:
        run_demo(pipeline)
    return 0


if __name__ == "__main__":
    exit(main())
