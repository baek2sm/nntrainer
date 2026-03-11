#ifndef VOCAB_SELECTION_H
#define VOCAB_SELECTION_H

#include <tensor.h>

#ifndef LSH_BLOCK_SIZE
#define LSH_BLOCK_SIZE 256
#endif

using namespace std;

namespace custom {

enum LshType { NONE = 0, SIMHASH = 1, ORTHOSIMHASH = 2 };
typedef std::bitset<LSH_BLOCK_SIZE> lshDataBlock;

class VocabSelection {
protected:
  size_t hiddenSize;
  size_t vocabCnt;
  const size_t lshBlockSize = LSH_BLOCK_SIZE;
  size_t lshBlockNum;
  size_t lshBits; // lshBlockSize * lshBlockNum
  size_t lshChoices;
  LshType lshType;
  std::vector<lshDataBlock> lshData;

public:
  VocabSelection(LshType lshType, size_t lshChoices, size_t hiddenSize,
                 size_t vocabCnt);
  virtual std::vector<std::vector<int>>
  getVocabs(const nntrainer::Tensor &modelOutput) = 0;
  virtual ~VocabSelection();
};

class VocabSelectionNNTrainer : public VocabSelection {
protected:
  nntrainer::Tensor lshWeight;

public:
  VocabSelectionNNTrainer(LshType lshType, size_t lshChoices, size_t hiddenSize,
                          size_t vocabCnt, nntrainer::Tensor &weights);
  std::vector<std::vector<int>> getVocabs(const nntrainer::Tensor &modelOutput);
  ~VocabSelectionNNTrainer(){};
};

} // namespace custom

#endif
