#include <gtest/gtest.h>
#include <multi_head_attention_layer.h>
#include <nntrainer_test_util.h>

/**
 * @brief Test fixture for MultiHeadAttentionLayer.
 */
class MultiHeadAttentionLayerTest : public ::testing::Test {
protected:
  void SetUp() override {
  }
  void TearDown() override {
  }
};

/**
 * @brief Simple test case for MHA layer initialization.
 */
TEST_F(MultiHeadAttentionLayerTest, InitializationTest) {
  nntrainer::MultiHeadAttentionLayer layer;
  // Initialize with some properties
  std::vector<std::string> props = {"num_heads=2", "projected_key_dim=4"};
  EXPECT_NO_THROW(layer.setProperty(props));
}
