// AVL Tree - Nick Thompson
// An implementation of a self-balancing binary search tree using AVL algorithm
//
// Copyright (c) 2026 Nick Thompson
// SPDX-License-Identifier: MIT

/// @file avl_tree.cpp
/// @brief Implementation of AVLTree class methods for self-balancing BST operations
#include "avl_tree/avl_tree.h"
#include <algorithm>
#include <optional>

AVLTree::Node::Node(std::string k, int v) : key(std::move(k)), value(v), height(1), left(nullptr), right(nullptr) {}

int AVLTree::_height(const Node* n) {
    return n ? n->height : 0;
}

void AVLTree::_fix(Node* n) {
    n->height = 1 + std::max(_height(n->left.get()), _height(n->right.get()));
}

int AVLTree::_bf(const Node* n) {
    return _height(n ? n->left.get() : nullptr) - _height(n ? n->right.get() : nullptr);
}

std::unique_ptr<AVLTree::Node> AVLTree::_rotate_right(std::unique_ptr<Node> z) {
    std::unique_ptr<Node> y = std::move(z->left);
    z->left = std::move(y->right);
    _fix(z.get());
    y->right = std::move(z);
    _fix(y.get());
    return y;
}

std::unique_ptr<AVLTree::Node> AVLTree::_rotate_left(std::unique_ptr<Node> z) {
    std::unique_ptr<Node> y = std::move(z->right);
    z->right = std::move(y->left);
    _fix(z.get());
    y->left = std::move(z);
    _fix(y.get());
    return y;
}

std::unique_ptr<AVLTree::Node> AVLTree::_rebalance(std::unique_ptr<Node> n) {
    int balance_factor = _bf(n.get());

    // Left-Left case: rotate right
    if (balance_factor > 1 && _bf(n->left.get()) >= 0) {
        return _rotate_right(std::move(n));
    }

    // Left-Right case: rotate left on left, then right
    if (balance_factor > 1 && _bf(n->left.get()) < 0) {
        n->left = _rotate_left(std::move(n->left));
        return _rotate_right(std::move(n));
    }

    // Right-Right case: rotate left
    if (balance_factor < -1 && _bf(n->right.get()) <= 0) {
        return _rotate_left(std::move(n));
    }

    // Right-Left case: rotate right on right, then left
    if (balance_factor < -1 && _bf(n->right.get()) > 0) {
        n->right = _rotate_right(std::move(n->right));
        return _rotate_left(std::move(n));
    }

    // Balanced
    return n;
}

std::unique_ptr<AVLTree::Node> AVLTree::_insert(std::unique_ptr<Node> n, const std::string& key, int value) {
    if (!n) {
        return std::make_unique<Node>(key, value);
    }

    if (key < n->key) {
        n->left = _insert(std::move(n->left), key, value);
    } else if (key > n->key) {
        n->right = _insert(std::move(n->right), key, value);
    } else {
        // Key already exists, update value
        n->value = value;
        return n;
    }

    _fix(n.get());
    return _rebalance(std::move(n));
}

void AVLTree::insert(const std::string& key, int value) {
    root_ = _insert(std::move(root_), key, value);
}

std::optional<int> AVLTree::search(const std::string& key) const {
    Node* current = root_.get();
    while (current) {
        if (key < current->key) {
            current = current->left.get();
        } else if (key > current->key) {
            current = current->right.get();
        } else {
            return current->value;
        }
    }
    return std::nullopt;
}

std::vector<std::pair<std::string, int>> AVLTree::in_order() const {
    std::vector<std::pair<std::string, int>> result;
    _in_order(root_.get(), result);
    return result;
}

void AVLTree::_in_order(const Node* n, std::vector<std::pair<std::string, int>>& result) const {
    if (!n) return;
    _in_order(n->left.get(), result);
    result.emplace_back(n->key, n->value);
    _in_order(n->right.get(), result);
}

int AVLTree::height() const {
    return _height(root_.get());
}
