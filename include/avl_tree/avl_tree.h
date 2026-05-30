// AVL Tree header file
#pragma once

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
    struct Node { };
};
