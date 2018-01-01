package me.peyppicp.cnn;

import com.google.common.base.Preconditions;
import com.vdurmont.emoji.EmojiParser;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * @author YuXiao Pan
 * @date 2017/12/28
 * @email yuxiao.pan@kikatech.com
 */
public class RnnTest {

    private final WordVectors wordVectors;
    private final WordToIndex wordToIndex;
    private final TokenizerFactory tokenizerFactory;
    private final MultiLayerNetwork ptbModel;
    private final ComputationGraph emojiModel;
    private final int word2vecSize;
    private final PTBEvaluation evaluation;
    private final int numberSteps;

    public RnnTest(WordVectors wordVectors, WordToIndex wordToIndex, TokenizerFactory tokenizerFactory, MultiLayerNetwork ptbModel, ComputationGraph emojiModel, int numberSteps) {
        this.wordVectors = wordVectors;
        this.wordToIndex = wordToIndex;
        this.tokenizerFactory = tokenizerFactory;
        this.ptbModel = ptbModel;
        this.emojiModel = emojiModel;
        this.word2vecSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
        this.evaluation = PTBEvaluation.getInstance();
        this.numberSteps = numberSteps;
    }

    public void generateTokensFromStr(String sentence, int topN) {
        ptbModel.rnnClearPreviousState();
        sentence = EmojiParser.removeAllEmojis(sentence);
        List<String> tokens = tokenizerFactory.create(sentence).getTokens();
        int threshold = tokens.size() - numberSteps;
        if (threshold < 0) {
            //直接预测
            for (int i = 0, j = 1; i < tokens.size() && j < tokens.size(); i++, j++) {
                String current = tokens.get(i);
                String next = tokens.get(j);
                INDArray zeros = Nd4j.zeros(1, word2vecSize);
                INDArray vectorMatrix = wordVectors.getWordVectorMatrix(current);
                zeros.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all()}, vectorMatrix);
                INDArray indArray = ptbModel.rnnTimeStep(zeros);
                List<String> top3words = findTopNWords(indArray, topN).stream().filter(s -> !"<end>".equals(s) && !"<unknown>".equals(s))
                        .limit(3)
                        .collect(Collectors.toList());
                if (top3words.get(0).equals(next)) {
                    evaluation.plusTop1Correct();
                } else {
                    evaluation.plusTop1Error();
                }
                if (top3words.contains(next)) {
                    evaluation.plusTop3Correct();
                } else {
                    evaluation.plusTop3Error();
                }
            }
        } else {
//            预测前numberStep个
            for (int i = 0; i < numberSteps; i++) {
                String current = tokens.get(i);
                String next = tokens.get(i);
                INDArray zeros = Nd4j.zeros(1, word2vecSize);
                INDArray vectorMatrix = wordVectors.getWordVectorMatrix(current);
                zeros.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all()}, vectorMatrix);
                INDArray indArray = ptbModel.rnnTimeStep(zeros);
                List<String> top3words = findTopNWords(indArray, topN).stream().filter(s -> !"<end>".equals(s) && !"<unknown>".equals(s))
                        .limit(3)
                        .collect(Collectors.toList());
                if (top3words.get(0).equals(next)) {
                    evaluation.plusTop1Correct();
                } else {
                    evaluation.plusTop1Error();
                }
                if (top3words.contains(next)) {
                    evaluation.plusTop3Correct();
                } else {
                    evaluation.plusTop3Error();
                }
            }
            for (int i = 1; i < tokens.size(); i++) {
                ptbModel.rnnClearPreviousState();
                if (i + numberSteps < tokens.size()) {
                    INDArray indArray = null;
                    for (int j = i; j < tokens.size(); j++) {
                        String preToken = tokens.get(j);
                        INDArray zeros = Nd4j.zeros(1, word2vecSize);
                        INDArray vectorMatrix = wordVectors.getWordVectorMatrix(preToken);
                        zeros.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all()}, vectorMatrix);
                        indArray = ptbModel.rnnTimeStep(zeros);
                    }
                    String current = tokens.get(i + numberSteps);
                }
            }
        }
        System.out.println("Top1:" + evaluation.getCorrectTop1Rate() + ", top3:" + evaluation.getCorrectTop3Rate());
    }

    public void prepareInputForPredict(List<String> input, String predictToken, int topN) {
        ptbModel.rnnClearPreviousState();
        INDArray indArray = null;
        for (int i = 0; i < input.size(); i++) {
            String previousToken = input.get(i);
            INDArray zeros = Nd4j.zeros(1, word2vecSize);
            INDArray previousVector = wordVectors.getWordVectorMatrix(previousToken);
            zeros.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all()}, previousVector);
            indArray = ptbModel.rnnTimeStep(zeros);
        }

        List<String> top3words = findTopNWords(indArray, topN).stream().filter(s -> !"<end>".equals(s) && !"<unknown>".equals(s))
                .limit(3)
                .collect(Collectors.toList());
        if (top3words.get(0).equals(predictToken)) {
            evaluation.plusTop1Correct();
        } else {
            evaluation.plusTop1Error();
        }
        if (top3words.contains(predictToken)) {
            evaluation.plusTop3Correct();
        } else {
            evaluation.plusTop3Error();
        }
    }

    public List<String> findTopNWords(INDArray output, int topN) {
        INDArray indArray = output.linearView();
        List<String> labels = wordToIndex.getLabels();
        Map<String, Float> top10WordsMap = new HashMap<>(15000);
        List<String> top10Words = new ArrayList<>(topN);
        Preconditions.checkArgument(topN < labels.size());
        Preconditions.checkArgument(labels.size() == indArray.length());
        for (int i = 0; i < indArray.length(); i++) {
            top10WordsMap.put(labels.get(i), indArray.getFloat(i));
        }
        top10WordsMap.entrySet().parallelStream()
                .sorted((o1, o2) -> -o1.getValue().compareTo(o2.getValue()))
                .limit(topN)
                .forEachOrdered(entry -> top10Words.add(entry.getKey()));
        return top10Words;
    }
}
