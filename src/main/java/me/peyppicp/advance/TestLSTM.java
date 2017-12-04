package me.peyppicp.advance;

import com.vdurmont.emoji.Emoji;
import com.vdurmont.emoji.EmojiManager;
import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class TestLSTM {

    public static void main(String[] args) throws IOException {

        int truncateReviewsToLength = 256;
        int vectorSize = 0;
        int batchSize = 150;
        int nEpochs = 1;
        String path = "distinctLines.txt";
        String labelPath = "commonLabelWithIndex.txt";
        String word2VecPath = "word2vecLookUpTable.txt";

        MultiLayerNetwork multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork("model-test01-4.txt");
        WordVectors wordVectors = WordVectorSerializer.readWord2VecModel(word2VecPath);
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();

        vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;

        Random random = new Random(123456);
        List<String> labels = FileUtils.readLines(new File(labelPath), Charsets.UTF_8);
        List<String> testLines = FileUtils.readLines(new File(path), Charsets.UTF_8);
        Collections.shuffle(testLines, random);
        Collections.shuffle(labels, random);
        List<String> subTestLines = testLines.subList(0, 1000);
        Map<String, String> map = new HashMap<>();

        for (String line : subTestLines) {
            String emojiStr = "";
            for (Emoji emoji : EmojiManager.getAll()) {
                if (line.contains(emoji.getUnicode())) {
                    line = line.replaceAll(emoji.getUnicode(), "");
                    emojiStr += emoji.getUnicode();
                }
            }
            map.put(line, emojiStr);
        }

        int labelSize = labels.parallelStream().collect(Collectors.toSet()).size();

        EDataSetIterator eDataSetIterator = new EDataSetIterator(path, labelPath, wordVectors, batchSize, truncateReviewsToLength, true);

        System.out.println("Begin test.");
        Evaluation evaluate = multiLayerNetwork.evaluate(eDataSetIterator);
        System.out.println(evaluate.stats());

//        for (Map.Entry<String, String> entry : map.entrySet()) {
//            String line = entry.getKey();
//            String emojiStr = entry.getValue();
//            List<String> tokens = tokenizerFactory.create(line).getTokens();
//            List<String> tokenFiltered = new ArrayList<>();
//            for (String t : tokens) {
//                if (word2Vec.hasWord(t)) {
//                    tokenFiltered.add(t);
//                }
//            }
//            int outputLength = Math.max(truncateReviewsToLength, tokenFiltered.size());
//            INDArray features = Nd4j.create(1, vectorSize, outputLength);
//            for (int j = 0; j < tokens.size() && j < truncateReviewsToLength; j++) {
//                String token = tokens.get(j);
//                INDArray vectorMatrix = word2Vec.getWordVectorMatrix(token);
//                features.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(j)}, vectorMatrix);
//            }
//
//            INDArray output = multiLayerNetwork.output(features);
//            int size = output.size(2);
//            INDArray probabilitiesAtLastWord = output.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(size - 1));
//
//            System.out.println("\n-------------------------------");
//            System.out.println("review: " + line);
//            System.out.println("Prefer:" + emojiStr);
//            System.out.println("label: " + probabilitiesAtLastWord.getDouble(1));
//        }
    }
}
