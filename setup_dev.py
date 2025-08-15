#!/usr/bin/env python3
"""Development environment setup script.

This script sets up a complete development environment for the Improved Local AI Assistant,
including all dependencies, pre-commit hooks, and validation checks.
"""

import subprocess
import sys
from pathlib import Path
from typing import List
from typing import Optional


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def run_command(
    cmd: List[str], check: bool = True, cwd: Optional[Path] = None
) -> subprocess.CompletedProcess:
    """Run a command and return the result.

    Args:
        cmd: Command to run as a list of strings
        check: Whether to raise an exception on non-zero exit code
        cwd: Working directory for the command

    Returns:
        CompletedProcess instance with the result

    Raises:
        subprocess.CalledProcessError: If check=True and command fails
    """
    print(f"{Colors.BLUE}Running: {' '.join(cmd)}{Colors.END}")
    return subprocess.run(cmd, check=check, cwd=cwd, capture_output=True, text=True)


def check_python_version() -> bool:
    """Check if Python version meets requirements.

    Returns:
        True if Python version is acceptable, False otherwise
    """
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(
            f"{Colors.RED}Error: Python 3.8+ required, found {version.major}.{version.minor}{Colors.END}"
        )
        return False

    print(
        f"{Colors.GREEN}✓ Python {version.major}.{version.minor}.{version.micro} detected{Colors.END}"
    )
    return True


def check_git() -> bool:
    """Check if git is available.

    Returns:
        True if git is available, False otherwise
    """
    try:
        result = run_command(["git", "--version"], check=False)
        if result.returncode == 0:
            print(f"{Colors.GREEN}✓ Git available{Colors.END}")
            return True
    except FileNotFoundError:
        pass

    print(f"{Colors.RED}Error: Git not found. Please install Git.{Colors.END}")
    return False


def setup_virtual_environment() -> bool:
    """Set up Python virtual environment.

    Returns:
        True if successful, False otherwise
    """
    venv_path = Path(".venv")

    if venv_path.exists():
        print(f"{Colors.YELLOW}Virtual environment already exists at {venv_path}{Colors.END}")
        return True

    try:
        print(f"{Colors.BLUE}Creating virtual environment...{Colors.END}")
        run_command([sys.executable, "-m", "venv", str(venv_path)])
        print(f"{Colors.GREEN}✓ Virtual environment created{Colors.END}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}Error creating virtual environment: {e}{Colors.END}")
        return False


def get_pip_executable() -> str:
    """Get the pip executable path for the virtual environment.

    Returns:
        Path to pip executable
    """
    if sys.platform == "win32":
        return str(Path(".venv") / "Scripts" / "pip.exe")
    else:
        return str(Path(".venv") / "bin" / "pip")


def install_dependencies() -> bool:
    """Install project dependencies.

    Returns:
        True if successful, False otherwise
    """
    pip_exe = get_pip_executable()

    try:
        print(f"{Colors.BLUE}Upgrading pip...{Colors.END}")
        run_command([pip_exe, "install", "--upgrade", "pip"])

        print(f"{Colors.BLUE}Installing development dependencies...{Colors.END}")
        run_command([pip_exe, "install", "-e", ".[dev]"])

        print(f"{Colors.GREEN}✓ Dependencies installed{Colors.END}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}Error installing dependencies: {e}{Colors.END}")
        return False


def setup_pre_commit() -> bool:
    """Set up pre-commit hooks.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Get the python executable from the virtual environment
        if sys.platform == "win32":
            python_exe = str(Path(".venv") / "Scripts" / "python.exe")
        else:
            python_exe = str(Path(".venv") / "bin" / "python")

        print(f"{Colors.BLUE}Installing pre-commit hooks...{Colors.END}")
        run_command([python_exe, "-m", "pre_commit", "install"])

        print(f"{Colors.GREEN}✓ Pre-commit hooks installed{Colors.END}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}Error setting up pre-commit: {e}{Colors.END}")
        return False


def run_initial_checks() -> bool:
    """Run initial code quality and test checks.

    Returns:
        True if all checks pass, False otherwise
    """
    if sys.platform == "win32":
        python_exe = str(Path(".venv") / "Scripts" / "python.exe")
    else:
        python_exe = str(Path(".venv") / "bin" / "python")

    checks = [
        ([python_exe, "-m", "ruff", "check", "."], "Ruff linting"),
        ([python_exe, "-m", "black", "--check", "."], "Black formatting"),
        ([python_exe, "-m", "mypy", "src/"], "MyPy type checking"),
    ]

    all_passed = True

    for cmd, description in checks:
        try:
            print(f"{Colors.BLUE}Running {description}...{Colors.END}")
            run_command(cmd)
            print(f"{Colors.GREEN}✓ {description} passed{Colors.END}")
        except subprocess.CalledProcessError:
            print(
                f"{Colors.YELLOW}⚠ {description} found issues (run 'make format' to fix){Colors.END}"
            )
            all_passed = False

    return all_passed


def print_next_steps() -> None:
    """Print next steps for the user."""
    print(f"\n{Colors.BOLD}{Colors.GREEN}Development environment setup complete!{Colors.END}")
    print(f"\n{Colors.BOLD}Next steps:{Colors.END}")

    if sys.platform == "win32":
        activate_cmd = ".venv\\Scripts\\activate"
    else:
        activate_cmd = "source .venv/bin/activate"

    print("1. Activate the virtual environment:")
    print(f"   {Colors.BLUE}{activate_cmd}{Colors.END}")
    print("\n2. Run the application:")
    print(f"   {Colors.BLUE}python run_app.py{Colors.END}")
    print("\n3. Run tests:")
    print(f"   {Colors.BLUE}make test{Colors.END}")
    print("\n4. Format code:")
    print(f"   {Colors.BLUE}make format{Colors.END}")
    print("\n5. Run all quality checks:")
    print(f"   {Colors.BLUE}make ci{Colors.END}")

    print(f"\n{Colors.BOLD}Available make commands:{Colors.END}")
    print(f"   {Colors.BLUE}make help{Colors.END} - Show all available commands")


def main() -> int:
    """Main setup function.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print(
        f"{Colors.BOLD}Setting up Improved Local AI Assistant development environment{Colors.END}\n"
    )

    # Check prerequisites
    if not check_python_version():
        return 1

    if not check_git():
        return 1

    # Setup steps
    steps = [
        (setup_virtual_environment, "Setting up virtual environment"),
        (install_dependencies, "Installing dependencies"),
        (setup_pre_commit, "Setting up pre-commit hooks"),
    ]

    for step_func, description in steps:
        print(f"\n{Colors.BOLD}{description}...{Colors.END}")
        if not step_func():
            print(f"{Colors.RED}Setup failed at: {description}{Colors.END}")
            return 1

    # Run initial checks (non-blocking)
    print(f"\n{Colors.BOLD}Running initial code quality checks...{Colors.END}")
    run_initial_checks()

    print_next_steps()
    return 0


if __name__ == "__main__":
    sys.exit(main())
