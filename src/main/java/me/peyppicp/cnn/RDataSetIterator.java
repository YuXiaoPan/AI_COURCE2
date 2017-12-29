package me.peyppicp.cnn;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author YuXiao Pan
 * @date 2017/12/17
 * @email yuxiao.pan@kikatech.com
 */
public class RDataSetIterator implements DataSetIterator {

    //    private final int linesToReadPerBatch;
    private final int batchSize;
    private final int truncateLength;
    private List<String> samples;
    private int cursor = 0;
    private final TokenizerFactory tokenizerFactory;
    private final WordToIndex wordToIndex;
    private final WordVectors wordVectors;
    private int vectorSize;

    public RDataSetIterator(boolean isTrain, int truncateLength, int batchSize,
                            List<String> samples, WordToIndex wordToIndex,
                            WordVectors wordVectors) {
//        this.linesToReadPerBatch = linesToReadPerBatch;
        this.truncateLength = truncateLength;
        this.samples = samples;
        this.batchSize = batchSize;
        this.tokenizerFactory = new DefaultTokenizerFactory();
        this.tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        this.wordToIndex = wordToIndex;
        this.wordVectors = wordVectors;
        this.vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
        if (isTrain) {
            samples = samples.subList(0, 25);
        }
        Collections.shuffle(samples);
    }

    @Override
    public DataSet next(int batchSize) {
        if (cursor >= samples.size()) {
            throw new RuntimeException();
        }

        int maxWordsSize = 0;
        List<List<String>> words = new ArrayList<>();
        for (int i = 0; i < batchSize && cursor < samples.size(); i++, cursor++) {
            String sample = samples.get(cursor);
            List<String> tokens = Arrays.stream(sample.split(" ")).collect(Collectors.toList());
            maxWordsSize = Math.max(maxWordsSize, tokens.size());
            words.add(tokens);
//            tokens.parallelStream().forEachOrdered(words::add);
//            words.add(tokens.parallelStream().collect(Collectors.toList()));
        }

        INDArray input = Nd4j.create(new int[]{batchSize,
                vectorSize, maxWordsSize}, 'f');
        INDArray labels = Nd4j.create(new int[]{batchSize,
                wordToIndex.getTotalWordsCount(), maxWordsSize}, 'f');
        INDArray inputMask = Nd4j.zeros(batchSize, maxWordsSize);
        INDArray labelMask = Nd4j.zeros(batchSize, maxWordsSize);

        for (int i = 0; i < words.size(); i++) {
            List<String> lineTokens = words.get(i);
            for (int j = 0; j < lineTokens.size(); j++) {
                String currentToken = lineTokens.get(j);
                INDArray currentVector = wordVectors.getWordVectorMatrix(currentToken);
                int timeStep = 0;
                for (int k = j + 1; k < maxWordsSize && k < lineTokens.size(); k++, timeStep++) {
                    String nextWord = lineTokens.get(k);
                    input.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(timeStep)}, currentVector);
                    labels.putScalar(new int[]{i, wordToIndex.getIndex(nextWord), timeStep}, 1.0);

                    currentVector = wordVectors.getWordVectorMatrix(nextWord);
                    currentToken = nextWord;
                    labelMask.putScalar(new int[]{i, timeStep}, 1.0);
                }
                break;
            }
            inputMask.putScalar(new int[]{i, 0}, 1.0);
        }
//        return new DataSet(input, labels, inputMask, labelMask);
        return new DataSet(input, labels, inputMask, labelMask);
    }

    @Override
    public int totalExamples() {
        return samples.size();
    }

    @Override
    public int inputColumns() {
        return vectorSize;
    }

    @Override
    public int totalOutcomes() {
        return wordToIndex.getTotalWordsCount();
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        cursor = 0;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        return cursor;
    }

    @Override
    public int numExamples() {
        return totalExamples();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> getLabels() {
        return wordToIndex.getLabels();
    }

    @Override
    public boolean hasNext() {
        return cursor < samples.size();
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }
}
