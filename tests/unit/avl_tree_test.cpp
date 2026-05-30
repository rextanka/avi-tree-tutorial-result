#include "avl_tree/avl_tree.h"
#include <gtest/gtest.h>

// Verifies that an empty AVLTree returns an empty vector from in_order()
TEST(AVLTreeTest, EmptyTreeReturnsEmptyInOrder) {
    AVLTree t;
    EXPECT_TRUE(t.in_order().empty());
}

// Verifies that in_order() returns all inserted pairs sorted by key
TEST(AVLTreeTest, InOrderReturnsSortedPairs) {
    AVLTree t;
    t.insert("c", 3);
    t.insert("a", 1);
    t.insert("b", 2);

    auto result = t.in_order();
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], std::make_pair("a", 1));
    EXPECT_EQ(result[1], std::make_pair("b", 2));
    EXPECT_EQ(result[2], std::make_pair("c", 3));
}

// Verifies that search() returns the correct value for existing keys
TEST(AVLTreeTest, SearchFindsExistingKey) {
    AVLTree t;
    t.insert("key1", 42);
    t.insert("key2", 99);

    auto result1 = t.search("key1");
    ASSERT_TRUE(result1.has_value());
    EXPECT_EQ(*result1, 42);

    auto result2 = t.search("key2");
    ASSERT_TRUE(result2.has_value());
    EXPECT_EQ(*result2, 99);
}

// Verifies that search() returns std::nullopt for keys not in the tree
TEST(AVLTreeTest, SearchReturnsNulloptForMissingKey) {
    AVLTree t;
    t.insert("existing", 100);

    auto result = t.search("not-present");
    EXPECT_FALSE(result.has_value());
}

// Verifies that inserting on an existing key updates the value
TEST(AVLTreeTest, InsertOnExistingKeyUpdatesValue) {
    AVLTree t;
    t.insert("a", 1);
    t.insert("a", 99);

    auto result = t.search("a");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 99);
}

// Verifies the tree remains balanced after inserting sorted keys
TEST(AVLTreeTest, BalancedAfterSortedInsert) {
    AVLTree t;
    // Insert keys "a" through "g" in alphabetical order
    for (char c = 'a'; c <= 'g'; ++c) {
        std::string key(1, c);
        t.insert(key, static_cast<int>(c));
    }

    // For a balanced tree of 7 nodes, height should be at most 4
    EXPECT_LE(t.height(), 4);
}