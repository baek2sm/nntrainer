#include "custom_vocab_selection.h"
#include <algorithm>

custom::VocabSelection::VocabSelection(LshType lshType, size_t lshChoices,
                                       size_t hiddenSize, size_t vocabCnt) :
  hiddenSize(hiddenSize),
  vocabCnt(vocabCnt),
  lshBlockNum(0),
  lshBits(0),
  lshChoices(lshChoices),
  lshType(lshType) {}

custom::VocabSelection::~VocabSelection() {}

custom::VocabSelectionNNTrainer::VocabSelectionNNTrainer(
  LshType lshType, size_t lshChoices, size_t hiddenSize, size_t vocabCnt,
  nntrainer::Tensor &weights) :
  VocabSelection(lshType, lshChoices, hiddenSize, vocabCnt) {
  this->lshBlockNum = (hiddenSize + lshBlockSize - 1) / lshBlockSize;
  this->lshBits = lshBlockNum * lshBlockSize;
  this->lshData = std::vector<lshDataBlock>(this->vocabCnt * lshBlockNum);

  // for (size_t i = 0; i < vocabCnt; ++i) {
  //     for (size_t j = 0; j < lshBlockNum; ++j) {
  //         size_t actualSize = std::min(lshBlockSize, hiddenSize - (int)j *
  //         lshBlockSize); lshDataBlock d; for (size_t k = 0; k < actualSize;
  //         ++k) {
  //             d[k] = weights.getValue<_FP16>(0, 0, i, j * lshBlockSize + k) >
  //             0 ? 1 : 0;
  //         }
  //         for (size_t k = actualSize; k < lshBlockSize; ++k) {
  //             d[k] = 0;
  //         }
  //         this->lshData[i * lshBlockNum + j] = d;
  //     }
  // }

  for (size_t i = 0; i < lshBlockNum; ++i) {
    size_t actualSize =
      std::min(lshBlockSize, hiddenSize - (int)i * lshBlockSize);
    for (size_t j = 0; j < vocabCnt; ++j) {
      lshDataBlock d;
      for (size_t k = 0; k < actualSize; ++k) {
        if (weights.getDataType() == nntrainer::TensorDim::DataType::FP32) {
          d[k] = weights.getValue(0, 0, i * lshBlockSize + k, j) > 0 ? 1 : 0;
        } else if (weights.getDataType() ==
                   nntrainer::TensorDim::DataType::FP16) {
          d[k] =
            weights.getValue<_FP16>(0, 0, i * lshBlockSize + k, j) > 0 ? 1 : 0;
        }
      }
      for (size_t k = actualSize; k < lshBlockSize; ++k) {
        d[k] = 0;
      }
      this->lshData[j * lshBlockNum + i] = d;
    }
  }
}

std::vector<std::vector<int>>
custom::VocabSelectionNNTrainer::getVocabs(const nntrainer::Tensor &input) {
  unsigned int batchSize = input.height();

  std::vector<std::vector<int>> res = std::vector<std::vector<int>>(batchSize);
  for (size_t i = 0; i < batchSize; i++) {
    std::vector<lshDataBlock> d(lshBlockNum);
    for (size_t k = 0; k < lshBlockNum; k++) {
      size_t actualSize = std::min(lshBlockSize, hiddenSize - k * lshBlockSize);
      for (size_t j = 0; j < actualSize; j++) {
        if (input.getDataType() == nntrainer::TensorDim::DataType::FP32) {
          d[k][j] = input.getValue(0, 0, i, j + k * lshBlockSize) >= 0 ? 1 : 0;
        } else if (input.getDataType() ==
                   nntrainer::TensorDim::DataType::FP16) {
          d[k][j] =
            input.getValue<_FP16>(0, 0, i, j + k * lshBlockSize) >= 0 ? 1 : 0;
        }
      }
      for (size_t j = actualSize; j < lshBlockSize; j++) {
        d[k][j] = 0;
      }
    }
    std::vector<int> simResult(vocabCnt, 0);
    std::vector<int> simCount(lshBits + 1, 0);
    for (size_t j = 0; j < vocabCnt; j++) {
      for (size_t k = 0; k < lshBlockNum; k++) {
        simResult[j] += (d[k] ^ lshData[j * lshBlockNum + k]).count();
      }
      simCount[simResult[j]]++;
    }
    int cut = lshBits + 1;
    int leftover = 0;
    size_t countSum = 0;
    for (size_t j = 0; j <= lshBits; j++) {
      countSum += simCount[j];
      if (countSum > lshChoices) {
        cut = j;
        leftover = simCount[j] - (countSum - lshChoices);
        break;
      }
    }
    std::vector<int> selectedVocabs(lshChoices);
    int pos = 0;
    for (size_t j = 0; j < vocabCnt; j++) {
      if (simResult[j] <= cut) {
        if (simResult[j] < cut) {
          selectedVocabs[pos] = j;
          pos++;
        } else if (leftover > 0) {
          selectedVocabs[pos] = j;
          pos++;
          leftover--;
        }
      }
    }
    res[i] = selectedVocabs;
  }
  return res;
}
