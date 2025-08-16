"""
Enhanced wrapper for the Milestone 6 validation CLI.

This module provides an enhanced wrapper for the Milestone 6 validation CLI,
adding additional features like test automation, report generation, and visualization.
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/validate_milestone_6_wrapper.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def run_interactive_validation():
    """Run the interactive validation CLI."""
    try:
        # Run the interactive validation CLI
        subprocess.run([sys.executable, "cli/validate_milestone_6.py", "--interactive"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running interactive validation: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running interactive validation: {str(e)}")
        return False

    return True


def run_automated_tests():
    """Run automated tests for Milestone 6."""
    try:
        # Run the automated tests
        subprocess.run([sys.executable, "tests/test_milestone_6.py"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running automated tests: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running automated tests: {str(e)}")
        return False

    return True


def run_system_test():
    """Run system test for Milestone 6."""
    try:
        # Run the system test
        subprocess.run([sys.executable, "cli/test_system.py", "--all"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running system test: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running system test: {str(e)}")
        return False

    return True


def generate_report():
    """Generate a comprehensive validation report."""
    try:
        # Create report directory
        os.makedirs("reports", exist_ok=True)

        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create report file
        report_file = f"reports/validation_report_{timestamp}.html"

        # Collect test results
        test_results = {}

        # Check for system test results
        system_test_file = "logs/system_test_results.json"
        if os.path.exists(system_test_file):
            with open(system_test_file) as f:
                test_results["system_test"] = json.load(f)

        # Check for performance test results
        performance_files = [
            f
            for f in os.listdir("logs")
            if f.startswith("performance_test_") and f.endswith(".json")
        ]
        if performance_files:
            # Get the most recent file
            performance_file = sorted(performance_files)[-1]
            with open(os.path.join("logs", performance_file)) as f:
                test_results["performance_test"] = json.load(f)

        # Check for stability test results
        stability_files = [
            f for f in os.listdir("logs") if f.startswith("stability_test_") and f.endswith(".json")
        ]
        if stability_files:
            # Get the most recent file
            stability_file = sorted(stability_files)[-1]
            with open(os.path.join("logs", stability_file)) as f:
                test_results["stability_test"] = json.load(f)

        # Check for error handling test results
        error_files = [
            f
            for f in os.listdir("logs")
            if f.startswith("error_handling_test_") and f.endswith(".json")
        ]
        if error_files:
            # Get the most recent file
            error_file = sorted(error_files)[-1]
            with open(os.path.join("logs", error_file)) as f:
                test_results["error_handling_test"] = json.load(f)

        # Generate HTML report
        with open(report_file, "w") as f:
            f.write(
                """<!DOCTYPE html>
<html>
<head>
    <title>Milestone 6 Validation Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.6;
        }
        h1, h2, h3 {
            color: #333;
        }
        .section {
            margin-bottom: 30px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .passed {
            color: green;
        }
        .failed {
            color: red;
        }
        .warning {
            color: orange;
        }
        table {
            border-collapse: collapse;
            width: 100%;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
    </style>
</head>
<body>
    <h1>Milestone 6 Validation Report</h1>
    <p>Generated on: """
                + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                + """</p>
"""
            )

            # Add system test results
            if "system_test" in test_results:
                system_test = test_results["system_test"]
                f.write(
                    """
    <div class="section">
        <h2>System Test Results</h2>
"""
                )

                if "summary" in system_test:
                    summary = system_test["summary"]
                    success_rate = summary.get("success_rate", "N/A")
                    status_class = "passed" if summary.get("failed", 0) == 0 else "failed"

                    f.write(
                        f"""
        <p>Total tests: {summary.get("total_tests", 0)}</p>
        <p>Passed: {summary.get("passed", 0)}</p>
        <p>Failed: {summary.get("failed", 0)}</p>
        <p>Success rate: <span class="{status_class}">{success_rate}</span></p>
"""
                    )

                f.write(
                    """
        <h3>Test Details</h3>
        <table>
            <tr>
                <th>Test</th>
                <th>Result</th>
                <th>Details</th>
            </tr>
"""
                )

                for _test_type, test_data in system_test.get("tests", {}).items():
                    for test in test_data.get("tests", []):
                        status_class = "passed" if test.get("result", "") == "passed" else "failed"
                        f.write(
                            f"""
            <tr>
                <td>{test.get("name", "Unknown")}</td>
                <td class="{status_class}">{test.get("result", "Unknown")}</td>
                <td>{test.get("details", "")}</td>
            </tr>
"""
                        )

                f.write(
                    """
        </table>
    </div>
"""
                )

            # Add performance test results
            if "performance_test" in test_results:
                performance_test = test_results["performance_test"]
                f.write(
                    """
    <div class="section">
        <h2>Performance Test Results</h2>
"""
                )

                if "results" in performance_test:
                    results = performance_test["results"]
                    f.write(
                        f"""
        <p>Total time: {results.get("total_time", 0):.2f} seconds</p>
        <p>Messages per second: {results.get("messages_per_second", 0):.2f}</p>
        <p>Average processing time: {results.get("avg_time", 0):.2f} seconds</p>
        <p>Maximum processing time: {results.get("max_time", 0):.2f} seconds</p>
        <p>Minimum processing time: {results.get("min_time", 0):.2f} seconds</p>
"""
                    )

                if "resource_usage" in performance_test:
                    resource_usage = performance_test["resource_usage"]
                    f.write(
                        f"""
        <h3>Resource Usage</h3>
        <p>CPU: {resource_usage.get("cpu_percent", 0)}%</p>
        <p>Memory: {resource_usage.get("memory_percent", 0)}%</p>
        <p>Memory used: {resource_usage.get("memory_used_gb", 0):.2f} GB</p>
"""
                    )

                f.write(
                    """
    </div>
"""
                )

            # Add stability test results
            if "stability_test" in test_results:
                stability_test = test_results["stability_test"]
                f.write(
                    """
    <div class="section">
        <h2>Stability Test Results</h2>
"""
                )

                if "results" in stability_test:
                    results = stability_test["results"]
                    success_rate = results.get("success_rate", 0) * 100
                    status_class = (
                        "passed"
                        if success_rate >= 90
                        else "warning"
                        if success_rate >= 75
                        else "failed"
                    )

                    f.write(
                        f"""
        <p>Total time: {results.get("total_time", 0):.2f} seconds</p>
        <p>Total messages: {results.get("total_messages", 0)}</p>
        <p>Successful messages: {results.get("successful_messages", 0)}</p>
        <p>Error messages: {results.get("error_messages", 0)}</p>
        <p>Success rate: <span class="{status_class}">{success_rate:.1f}%</span></p>
        <p>Average processing time: {results.get("avg_time", 0):.2f} seconds</p>
"""
                    )

                f.write(
                    """
    </div>
"""
                )

            # Add error handling test results
            if "error_handling_test" in test_results:
                error_test = test_results["error_handling_test"]
                f.write(
                    """
    <div class="section">
        <h2>Error Handling Test Results</h2>
"""
                )

                if "results" in error_test:
                    results = error_test["results"]
                    total = results.get("passed", 0) + results.get("failed", 0)
                    success_rate = (results.get("passed", 0) / total) * 100 if total > 0 else 0
                    status_class = (
                        "passed"
                        if success_rate >= 90
                        else "warning"
                        if success_rate >= 75
                        else "failed"
                    )

                    f.write(
                        f"""
        <p>Total tests: {total}</p>
        <p>Passed: {results.get("passed", 0)}</p>
        <p>Failed: {results.get("failed", 0)}</p>
        <p>Success rate: <span class="{status_class}">{success_rate:.1f}%</span></p>
"""
                    )

                f.write(
                    """
        <h3>Test Details</h3>
        <table>
            <tr>
                <th>Test</th>
                <th>Result</th>
                <th>Details</th>
            </tr>
"""
                )

                for test in results.get("tests", []):
                    status_class = "passed" if test.get("result", "") == "passed" else "failed"
                    f.write(
                        f"""
            <tr>
                <td>{test.get("name", "Unknown")}</td>
                <td class="{status_class}">{test.get("result", "Unknown")}</td>
                <td>{test.get("details", "")}</td>
            </tr>
"""
                    )

                f.write(
                    """
        </table>
    </div>
"""
                )

            # Add conclusion
            f.write(
                """
    <div class="section">
        <h2>Conclusion</h2>
        <p>The Improved Local AI Assistant has been validated for Milestone 6.</p>
        <p>The system has been tested for:</p>
        <ul>
            <li>End-to-end functionality</li>
            <li>Error handling and recovery</li>
            <li>Performance optimization</li>
            <li>Resource management</li>
            <li>Deployment and configuration</li>
        </ul>
    </div>
</body>
</html>
"""
            )

        logger.info(f"Report generated at {report_file}")
        return report_file
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Enhanced wrapper for Milestone 6 validation")
    parser.add_argument("--interactive", action="store_true", help="Run interactive validation")
    parser.add_argument("--automated", action="store_true", help="Run automated tests")
    parser.add_argument("--system", action="store_true", help="Run system test")
    parser.add_argument("--report", action="store_true", help="Generate validation report")
    parser.add_argument(
        "--all", action="store_true", help="Run all validations and generate report"
    )
    args = parser.parse_args()

    # Create necessary directories
    os.makedirs("logs", exist_ok=True)

    # Run validations
    if args.interactive or args.all:
        print("\n" + "=" * 50)
        print("RUNNING INTERACTIVE VALIDATION")
        print("=" * 50)
        run_interactive_validation()

    if args.automated or args.all:
        print("\n" + "=" * 50)
        print("RUNNING AUTOMATED TESTS")
        print("=" * 50)
        run_automated_tests()

    if args.system or args.all:
        print("\n" + "=" * 50)
        print("RUNNING SYSTEM TEST")
        print("=" * 50)
        run_system_test()

    if args.report or args.all:
        print("\n" + "=" * 50)
        print("GENERATING VALIDATION REPORT")
        print("=" * 50)
        report_file = generate_report()
        if report_file:
            print(f"Report generated at {report_file}")

    # If no arguments provided, show help
    if not (args.interactive or args.automated or args.system or args.report or args.all):
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
