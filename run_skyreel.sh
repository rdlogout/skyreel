#!/bin/bash

# SkyReels-V2 Quick Launch Script
# This is a simple wrapper around the complete setup script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${CYAN}$1${NC}"
    echo -e "${CYAN}$(printf '=%.0s' {1..60})${NC}"
}

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    echo "🎬 SkyReels-V2 Quick Launch Script"
    echo "=================================="
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  setup       Complete setup (install dependencies, check GPU, etc.)"
    echo "  run         Setup and run with public sharing (default)"
    echo "  local       Setup and run locally only"
    echo "  test        Test current installation"
    echo "  deps        Install dependencies only"
    echo "  help        Show this help"
    echo ""
    echo "Options:"
    echo "  --port PORT     Port to run on (default: 7860)"
    echo "  --skip-deps     Skip dependency installation"
    echo ""
    echo "Examples:"
    echo "  $0                    # Complete setup and run with sharing"
    echo "  $0 local              # Run locally without sharing"
    echo "  $0 setup              # Setup only, don't run"
    echo "  $0 run --port 8080    # Run on port 8080"
    echo "  $0 deps               # Install dependencies only"
    echo ""
    echo "For RTX 6000 Blackwell users:"
    echo "  This script automatically detects your GPU and installs"
    echo "  PyTorch nightly with sm_120 support if needed."
}

# Check if Python script exists
check_script() {
    if [ ! -f "setup_and_run_complete.py" ]; then
        print_error "setup_and_run_complete.py not found!"
        print_error "Please make sure you're in the SkyReels-V2 directory"
        exit 1
    fi
}

# Source environment variables if they exist
source_env() {
    if [ -f "skyreel_env.sh" ]; then
        print_status "Loading environment variables..."
        source skyreel_env.sh
    fi
}

# Main function
main() {
    print_header "🎬 SkyReels-V2 Quick Launch"
    
    # Check if we have the main script
    check_script
    
    # Source environment variables
    source_env
    
    # Parse command
    case "${1:-run}" in
        "setup")
            print_status "Running complete setup..."
            python3 setup_and_run_complete.py --setup-only "${@:2}"
            ;;
        "run")
            print_status "Running complete setup and starting with public sharing..."
            python3 setup_and_run_complete.py "${@:2}"
            ;;
        "local")
            print_status "Running complete setup and starting locally..."
            python3 setup_and_run_complete.py --no-share "${@:2}"
            ;;
        "test")
            print_status "Testing current installation..."
            python3 setup_and_run_complete.py --setup-only --skip-deps "${@:2}"
            ;;
        "deps")
            print_status "Installing dependencies only..."
            python3 setup_and_run_complete.py --setup-only "${@:2}"
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}[INFO]${NC} Shutting down gracefully..."; exit 0' INT

main "$@"
