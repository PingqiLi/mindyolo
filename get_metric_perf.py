import re
import argparse


def extract_step_times(file_path):
    # Define the regex pattern to capture the step times
    pattern = r"step time:\s*(\d+\.\d+)\s*ms"

    step_times = []

    # Read the file and search for the pattern
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                step_times.append(float(match.group(1)))

    return step_times[1:]


def calculate_average_time(step_times):
    if len(step_times) == 0:
        return 0
    return sum(step_times) / len(step_times)


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Process step times from a text file.')
    parser.add_argument('file_path', type=str, help='Path to the text file')

    # Parse the arguments
    args = parser.parse_args()

    # Extract step times from the file
    step_times = extract_step_times(args.file_path)

    # Calculate the average time
    average_time = calculate_average_time(step_times)

    # Print the result
    print(f"Average step time: {average_time:.2f}ms")


if __name__ == "__main__":
    main()
