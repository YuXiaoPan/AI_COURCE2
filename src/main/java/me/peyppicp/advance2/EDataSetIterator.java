package me.peyppicp.advance2;

import com.google.common.io.Files;
import org.apache.commons.io.Charsets;
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

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * @author YuXiao Pan
 * @date 2017/11/27
 * @email yuxiao.pan@kikatech.com
 */
public class EDataSetIterator implements DataSetIterator {

    private final WordVectors wordVectors;
    private final int batchSize;
    private final int vectorSize;
    private final int truncateLength;
    private int maxLength;
    private final File file;
    private final File labelFile;
    private final boolean isTest;
    private int cursor = 0;
    private int currentLineCursor = 0;
    private List<String> totalLines;
    private List<String> totalLabelLinesWithIndex;
    private final TokenizerFactory tokenizerFactory;
    private WordToIndex wordToIndex;

    public EDataSetIterator(String sampleFilePath, String path, String labelPath,
                            String wordVectorPath, int batchSize,
                            int truncateLength, boolean isTest) throws IOException {
        this.wordToIndex = new WordToIndex(sampleFilePath, wordVectorPath);
        this.wordVectors = wordToIndex.getWordVectors();
        this.batchSize = batchSize;
        this.truncateLength = truncateLength;
        this.isTest = isTest;
        this.vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
        this.tokenizerFactory = new DefaultTokenizerFactory();
        this.tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        this.file = new File(path);
        this.labelFile = new File(labelPath);

        this.totalLines = Files.readLines(file, Charsets.UTF_8);
        this.totalLabelLinesWithIndex = Files.readLines(labelFile, Charsets.UTF_8);
        if (isTest) {
            Random random = new Random();
            Collections.shuffle(this.totalLines, random);
            Collections.shuffle(this.totalLines, random);
            this.totalLabelLinesWithIndex = this.totalLabelLinesWithIndex.subList(0, 5000);
            this.totalLines = this.totalLines.subList(0, 5000);
        }
    }

    public DataSet next(int batchSize) {
        if (cursor >= totalLines.size()) {
            throw new NoSuchElementException();
        }
        List<String> reviews = new ArrayList<>(batchSize);
        int[] labelInts = new int[batchSize];
        for (int i = 0; i < batchSize && cursor < totalExamples(); i++, cursor++, currentLineCursor++) {
            String line = totalLines.get(currentLineCursor);
            reviews.add(line);
            labelInts[i] = Integer
                    .parseInt(totalLabelLinesWithIndex
                            .get(currentLineCursor)
                            .split(",")[0]
                            .split("-")[1]);
        }
        List<List<String>> allTokens = new ArrayList<>();
        maxLength = 0;
        for (String line : reviews) {
            List<String> tokens = tokenizerFactory.create(line).getTokens();
            List<String> tokenFiltered = new ArrayList<>();
            for (String t : tokens) {
                if (wordVectors.hasWord(t)) {
                    tokenFiltered.add(t);
                }
            }
            allTokens.add(tokenFiltered);
            maxLength = Math.max(maxLength, tokenFiltered.size());
        }

        if (maxLength > truncateLength) {
            maxLength = truncateLength;
        }
        INDArray features = Nd4j.create(new int[]{reviews.size(), vectorSize, maxLength}, 'f');
        INDArray labels = Nd4j.create(new int[]{reviews.size(), wordToIndex.getOutComesNum(), maxLength}, 'f');
        INDArray featureMask = Nd4j.zeros(reviews.size(), maxLength);
        INDArray labelMask = Nd4j.zeros(reviews.size(), maxLength);

//        int[] ints = new int[maxLength];
        int[] ints = new int[2];
        for (int i = 0; i < reviews.size(); i++) {
            ints[0] = i;
            List<String> tokens = allTokens.get(i);
            for (int j = 0; j < tokens.size() && j < maxLength; j++) {
                String token = tokens.get(j);
                INDArray vector = wordVectors.getWordVectorMatrix(token);
                features.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);
                ints[1] = j;
                featureMask.putScalar(ints, 1.0);
            }
            int index = labelInts[i];
            int lastIndex = Math.min(tokens.size(), maxLength);
            labels.putScalar(new int[]{i, index, lastIndex - 1}, 1.0);
            labelMask.putScalar(new int[]{i, lastIndex - 1}, 1.0);
        }
        return new DataSet(features, labels, featureMask, labelMask);
    }

    public int totalExamples() {
        return totalLines.size();
    }

    public int inputColumns() {
        return vectorSize;
    }

    public int totalOutcomes() {
        return wordToIndex.getOutComesNum();
    }

    public boolean resetSupported() {
        return true;
    }

    public boolean asyncSupported() {
        return true;
    }

    public void reset() {
        this.cursor = 0;
        this.currentLineCursor = 0;
    }

    public int batch() {
        return batchSize;
    }

    public int cursor() {
        return cursor;
    }

    public int numExamples() {
        return totalExamples();
    }

    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    public List<String> getLabels() {
        return wordToIndex.totalLabels();
    }

    public boolean hasNext() {
        return cursor < numExamples();
    }

    public DataSet next() {
        return next(batchSize);
    }
}
