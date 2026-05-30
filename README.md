# AVL Tree C++ Project

A self-balancing binary search tree implementation in modern C++20, developed as part of the tutorial at `docs/local-llm-cline-tutorial.md`. This is a worked example of the tutorial at <link>.

## Overview

This project implements an **AVL Tree** - a height-balanced binary search tree that automatically maintains balance factors to ensure O(log n) operations for insertion, deletion, and search. The implementation uses C++20 features including structured bindings, `std::optional`, and smart pointers for memory safety.

## Development Environment

- **Hardware**: 48GB M4 Pro MacBook Pro
- **IDE**: Visual Studio Code with Cline plugin
- **LLM Model**: Qwen3.5:35b-a3b-coding-nvfp4 (hosted via Ollama, MLX optimized)
- **Build System**: CMake 3.24+

## Project Structure

```
avl-tree-cpp/
├── app/                    # Application source code
│   ├── main.cpp           # Entry point demonstrating AVLTree usage
│   └── CMakeLists.txt
├── include/avl_tree/      # Public headers
│   └── avl_tree.h         # AVLTree class interface
├── src/                   # Library implementation
│   ├── avl_tree.cpp       # AVLTree method implementations
│   └── CMakeLists.txt
├── tests/
│   ├── unit/              # Unit tests (GoogleTest)
│   │   ├── avl_tree_test.cpp
│   │   └── CMakeLists.txt
│   └── functional/        # Functional tests (pytest)
│       └── test_app.py
├── docs/                  # Documentation and tutorials
│   └── local-llm-cline-tutorial.md
└── CMakeLists.txt         # Root CMake configuration
```

## Features

- **Self-Balancing**: Automatically maintains height balance using rotations (left/right)
- **Modern C++20**: Uses structured bindings, `std::optional`, smart pointers
- **Memory Safe**: No manual memory management - uses `std::unique_ptr`
- **Comprehensive Testing**: Both unit tests and functional integration tests

## Building

```bash
# Configure and build
cmake --build build/

# Run unit tests
ctest --test-dir build/ --output-on-failure
```

## Running the Application

```bash
./build/app/avl_tree_app
```

Expected output:
```
apple: 5
apricot: 3
banana: 2
cherry: 1
date: 8
search cherry: 1
search mango: not found
height: 3
```

## Testing

### Unit Tests (GoogleTest)
Tests the AVLTree class directly through its public API.

```bash
ctest --test-dir build/ -R avl_tree_unit_tests
```

### Functional Tests (pytest)
Tests the compiled binary's behavior by invoking it and checking output.

```bash
cd tests/functional && pytest test_app.py
```

## License

Copyright (c) 2026 Nick Thompson  
SPDX-License-Identifier: MIT

See individual source files for full license text.

## Implementation Details

### AVL Tree Properties

- **Balance Factor**: Difference between left and right subtree heights (-1, 0, or +1)
- **Rotations**: Left rotation and right rotation to maintain balance
- **Height Bound**: For n nodes, height ≤ ⌊log₂(n)⌋ + 1

### Public API

```cpp
class AVLTree {
public:
    void insert(const std::string& key, int value);           // Insert key-value pair
    std::optional<int> search(const std::string& key) const; // Search by key
    std::vector<std::pair<std::string, int>> in_order() const; // Sorted traversal
    int height() const;                                       // Get tree height
};
```

## Performance Notes

The development process demonstrated efficient use of local LLM inference:
- Peak VRAM usage: ~31 GB (within 48GB unified memory limits)
- Context caching enabled rapid iteration with <8s response times for most prompts
- Large context ingestion (~9000 tokens) took ~20 seconds but was cached for subsequent queries

## Author

**Nick Thompson** - Developed using AI-assisted coding with Cline and Qwen3.5:35b-a3b-coding-nvfp4 model via Ollama.
