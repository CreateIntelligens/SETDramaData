#!/bin/bash

# Common utilities for Breeze ASR interactive script
# Functions shared across different modules

# Function to detect available Python command
detect_python() {
    # Allow override via environment variable
    if [ -n "$PYTHON_CMD" ]; then
        echo "$PYTHON_CMD"
        return 0
    fi
    
    # Auto-detect based on environment
    # Check if we're in Docker/container (common indicators)
    if [ -f "/.dockerenv" ] || [ -n "$DOCKER_CONTAINER" ]; then
        # In container - prefer direct python
        if command -v python &> /dev/null; then
            echo "python"
        elif command -v python3 &> /dev/null; then
            echo "python3"
        else
            echo ""
        fi
    # Check if poetry is available and we're in a poetry project
    elif command -v poetry &> /dev/null && [ -f "pyproject.toml" ]; then
        echo "poetry run python"
    # Fallback to system python
    elif command -v python3 &> /dev/null; then
        echo "python3"
    elif command -v python &> /dev/null; then
        echo "python"
    else
        echo ""
    fi
}

# Function to run Python script with proper command
run_python_script() {
    local script_file="$1"
    local python_cmd=$(detect_python)
    
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到 Python"
        return 1
    fi
    
    $python_cmd "$script_file"
    local result=$?
    rm -f "$script_file"
    return $result
}

# Function to validate episode number input
validate_episode_input() {
    local input="$1"
    local episodes=()
    
    if [ -z "$input" ]; then
        echo "❌ 請輸入集數"
        return 1
    fi
    
    # Parse input (range or list)
    if [[ "$input" =~ ^[0-9]+-[0-9]+$ ]]; then
        # Range format: 2-6
        local start=$(echo "$input" | cut -d'-' -f1)
        local end=$(echo "$input" | cut -d'-' -f2)
        for ((i=start; i<=end; i++)); do
            episodes+=("$i")
        done
    else
        # Single or multiple episodes: 1 or 1 3 5
        for num in $input; do
            if [[ "$num" =~ ^[0-9]+$ ]]; then
                episodes+=("$num")
            else
                echo "❌ 無效的集數: $num"
                return 1
            fi
        done
    fi
    
    if [ ${#episodes[@]} -eq 0 ]; then
        echo "❌ 沒有有效的集數"
        return 1
    fi
    
    # Export episodes array for caller
    printf '%s\n' "${episodes[@]}"
    return 0
}

# Function to pause for user input
pause_for_input() {
    local message="${1:-按 Enter 繼續...}"
    read -p "$message"
}

# Function to get confirmation from user
get_confirmation() {
    local message="$1"
    local response
    
    echo -n "$message [y/N]: "
    read response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

# Function to check if directory exists and is not empty
check_directory() {
    local dir="$1"
    
    if [ ! -d "$dir" ]; then
        echo "❌ 目錄不存在: $dir"
        return 1
    fi
    
    return 0
}

# Function to safely remove directory if empty
cleanup_empty_directory() {
    local dir="$1"
    
    if [ -d "$dir" ] && [ -z "$(ls -A "$dir" 2>/dev/null)" ]; then
        rmdir "$dir" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "✅ 已清理空目錄: $dir"
        fi
    fi
}