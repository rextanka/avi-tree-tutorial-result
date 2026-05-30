// AVL Tree header file
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

class AVLTree {
public:
    void insert(const std::string& key, int value);
    std::optional<int> search(const std::string& key) const;
    std::vector<std::pair<std::string, int>> in_order() const;
    int height() const;

private:
    struct Node {
        std::string key;
        int value;
        int height{1};
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        Node(std::string k, int v);
    };

    static int _height(const Node* n);
    static int _bf(const Node* n);
    static std::unique_ptr<Node> _rotate_right(std::unique_ptr<Node> z);
    static std::unique_ptr<Node> _rotate_left(std::unique_ptr<Node> z);
    static std::unique_ptr<Node> _rebalance(std::unique_ptr<Node> n);
    static void _fix(Node* n);
    static std::unique_ptr<Node> _insert(std::unique_ptr<Node> n, const std::string& key, int value);
    void _in_order(const Node* n, std::vector<std::pair<std::string, int>>& result) const;

    std::unique_ptr<Node> root_;
};
