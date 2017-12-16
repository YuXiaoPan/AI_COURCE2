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

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

public class TestLSTM {

    public static void main(String[] args) throws IOException {

        int truncateReviewsToLength = 256;
        int vectorSize = 0;
        int batchSize = 150;
        int nEpochs = 1;
        String path = "EmojiSampleWithoutEmoji.txt";
        String labelPath = "EmojiSampleLabels.txt";
        String word2VecPath = "LookUpTable.txt";
        String totalExamplesPath = "emoji_sample.txt";

        MultiLayerNetwork multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork("model-main01-12.txt");
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

        int testDataSize = 1000;

        InputStreamReader inputStreamReader = new InputStreamReader(new FileInputStream(new File(totalExamplesPath)), Charsets.UTF_8);
        BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
        List<String> testLines1 = new ArrayList<>();
        String str = "";
        int count = 0;
        while ((str = bufferedReader.readLine()) != null) {
            if (count <= testDataSize) {
                testLines1.add(str);
                count++;
            } else {
                break;
            }
        }
        bufferedReader.close();
        inputStreamReader.close();

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

//        for (String line : testLines1) {
//            List<String> tokens = tokenizerFactory.create(line).getTokens();
//            List<String> tokenFiltered = new ArrayList<>();
//            for (String t : tokens) {
//                if (wordVectors.hasWord(t)) {
//                    tokenFiltered.add(t);
//                }
//            }
//            int outputLength = Math.max(truncateReviewsToLength, tokenFiltered.size());
//            INDArray features = Nd4j.create(1, vectorSize, outputLength);
//            for (int j = 0; j < tokens.size() && j < truncateReviewsToLength; j++) {
//                String token = tokens.get(j);
//                INDArray vectorMatrix = wordVectors.getWordVectorMatrix(token);
//                features.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(j)}, vectorMatrix);
//            }
//
//            INDArray output = multiLayerNetwork.output(features, false);
//            int size = output.size(2);
//            System.out.println();
//            INDArray probabilitiesAtLastWord = output.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(size - 1));
//
//            Number number = probabilitiesAtLastWord.maxNumber();
//            long l = number.longValue();
//            List<String> unqiueLabelList = eDataSetIterator.getUnqiueLabelList();
//            System.out.println(probabilitiesAtLastWord.toString());
//            System.out.println("\n-------------------------------");
//            System.out.println("review: " + line);
//            System.out.println("Prefer:" + emojiStr);
//            System.out.println("label: " + probabilitiesAtLastWord.getDouble(1));
//        }
    }
}