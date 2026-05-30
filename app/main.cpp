// AVL Tree - Nick Thompson
// An implementation of a self-balancing binary search tree using AVL algorithm
//
// Copyright (c) 2026 Nick Thompson
// SPDX-License-Identifier: MIT

/// @file main.cpp
/// @brief Main application entry point demonstrating AVLTree usage
#include "avl_tree/avl_tree.h"
#include <iostream>
#include <cstdlib>

int main() {
    AVLTree tree;
    
    // Insert in specified order
    tree.insert("banana", 2);
    tree.insert("apple", 5);
    tree.insert("cherry", 1);
    tree.insert("date", 8);
    tree.insert("apricot", 3);
    
    // Iterate and print each pair
    for (const auto& [key, value] : tree.in_order()) {
        std::cout << key << ": " << value << "\n";
    }
    
    // Search for cherry
    auto cherry_result = tree.search("cherry");
    if (cherry_result.has_value()) {
        std::cout << "search cherry: " << *cherry_result << "\n";
    }
    
    // Search for mango (not found)
    auto mango_result = tree.search("mango");
    if (!mango_result.has_value()) {
        std::cout << "search mango: not found\n";
    }
    
    // Print height
    std::cout << "height: " << tree.height() << "\n";
    
    std::cout << std::flush;
    return EXIT_SUCCESS;
}