import os
import asyncio
import logging
import argparse
from dotenv import load_dotenv

# Bootstrap local environment secrets from the .env file
load_dotenv()

from core import LLMModelFactory, LLMModelConfig

# Setup clean, structured telemetry logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def parse_command_line_arguments():
    """Parses incoming terminal flags into typed configuration parameters."""
    parser = argparse.ArgumentParser(
        description="Enterprise Multi-LLM Factory CLI Execution Utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-p", "--provider",
        type=str,
        required=True,
        choices=["openai", "google", "aws_bedrock"],
        help="Target cloud interface provider infrastructure mapping"
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Target model deployment identifier string (e.g., 'gpt-4o', 'gemini-2.5-flash')"
    )
    
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature constraint modifier bound between 0.0 and 2.0"
    )
    
    parser.add_argument(
        "-f", "--prompt-file",
        type=str,
        default=None,
        help="Optional path to a local text file (.txt) containing the execution prompt payload"
    )

    parser.add_argument(
        "-o", "--output-file",
        type=str,
        default="output.txt",
        help="Target local path to save the generated text payload response"
    )
    
    return parser.parse_args()

async def main():
    # 1. Capture user inputs from the terminal command execution
    args = parse_command_line_arguments()
    
    print(f"\n=== Initializing Engine Connection Framework ===")
    print(f"Targeting Provider : {args.provider}")
    print(f"Targeting Model ID : {args.model}")
    print(f"Setting Temperature: {args.temperature}")
    print(f"Saving Output To   : {args.output_file}")
    if args.prompt_file:
        print(f"Reading Prompt From: {args.prompt_file}\n")
    else:
        print("Reading Prompt From: Default Fallback Context\n")

    try:
        # 2. Resolve prompt text from file or fall back to system default
        if args.prompt_file:
            if not os.path.exists(args.prompt_file):
                raise FileNotFoundError(f"The specified prompt file target was not found: '{args.prompt_file}'")
            
            with open(args.prompt_file, "r", encoding="utf-8") as file_stream:
                prompt_text = file_stream.read().strip()
                
            if not prompt_text:
                raise ValueError(f"The prompt file '{args.prompt_file}' is completely empty.")
        else:
            # High-level developer fallback default if the optional flag is omitted
            prompt_text = "Explain the core benefit of the Strategy Design Pattern in exactly two sentences."

        # 3. Package parsed CLI parameters straight into the Pydantic validator model
        config = LLMModelConfig(
            provider=args.provider,
            model_id=args.model,
            temperature=args.temperature
        )

        # 4. Request appropriate driver routing client from the Factory layer
        client = LLMModelFactory.create_client(config)

        print(f"Resolved Prompt Payload:\n\"\"\"\n{prompt_text}\n\"\"\"\n")
        print("Waiting for cloud API runtime return state...")
        
        # 5. Trigger the target execution pipeline asynchronously
        response = await client.agenerate(prompt=prompt_text)

        # 6. Output response metrics and telemetry back to terminal screen
        print("\n----------------- EXECUTION COMPLETED SUCCESSFULLY -----------------")
        print(f"Raw Output Text Content:\n{response.content}")
        print(f"\nExecution Footprint Metadata Tracker:\n{response.metrics.model_dump()}")
        print("--------------------------------------------------------------------")

        # 7. Write the model response content payload to the designated output target
        with open(args.output_file, "w", encoding="utf-8") as output_stream:
            output_stream.write(response.content)
        print(f"📝 Output successfully tracked and written to: '{args.output_file}'\n")

    except FileNotFoundError as fnf_err:
        logging.error(f"File IO targeted route error: {fnf_err}")
    except ValueError as val_err:
        logging.error(f"Configuration parameter mapping validation error: {val_err}")
    except Exception as runtime_err:
        logging.error(f"A catastrophic pipeline execution failure occurred: {runtime_err}")

if __name__ == "__main__":
    # Standard terminal application execution runtime container entry point
    asyncio.run(main())