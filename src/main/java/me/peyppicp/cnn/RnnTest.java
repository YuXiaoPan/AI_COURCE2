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

    public RnnTest(WordVectors wordVectors, WordToIndex wordToIndex, TokenizerFactory tokenizerFactory, MultiLayerNetwork ptbModel, ComputationGraph emojiModel) {
        this.wordVectors = wordVectors;
        this.wordToIndex = wordToIndex;
        this.tokenizerFactory = tokenizerFactory;
        this.ptbModel = ptbModel;
        this.emojiModel = emojiModel;
        this.word2vecSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
        this.evaluation = PTBEvaluation.getInstance();
    }

    public void generateTokensFromStr(String sentence, int topN) {
        ptbModel.rnnClearPreviousState();
        sentence = EmojiParser.removeAllEmojis(sentence);
        List<String> tokens = tokenizerFactory.create(sentence).getTokens();
        if (tokens.size() <= 1) {
//            top1.plusError();
//            top3.plusError();
            return;
        }
        for (int i = 0, j = 1; i < tokens.size() && j < tokens.size(); i++, j++) {
            String currentToken = tokens.get(i);
            String nextToken = tokens.get(j);
            INDArray zeros = Nd4j.zeros(1, word2vecSize);
            INDArray currentVector = wordVectors.getWordVectorMatrix(currentToken);
            zeros.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all()}, currentVector);
            INDArray output = ptbModel.rnnTimeStep(zeros);
            List<String> top3Words = findTopNWords(output, topN).stream().limit(3).collect(Collectors.toList());
            if (top3Words.contains(nextToken)) {
                evaluation.plusTop3Current();
            } else {
                evaluation.plusTop3Error();
            }

            if (top3Words.get(0).equals(nextToken)) {
                evaluation.plusTop1Current();
            } else {
                evaluation.plusTop1Error();
            }
        }
        System.out.println("Top1:" + evaluation.getCorrectTop1Rate() + ",Top3:" + evaluation.getCorrectTop3Rate());

//        for (String token : tokens) {
//            INDArray input = Nd4j.zeros(1, word2vecSize);
//            INDArray vectorMatrix = wordVectors.getWordVectorMatrix(token);
//            input.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all()}, vectorMatrix);
//            INDArray output = ptbModel.rnnTimeStep(input);
//            System.out.println("current input:" + token);
//            System.out.println(findTopNWords(output, topN));
//        }
//        System.out.println();
    }

    public List<String> findTopNWords(INDArray output, int topN) {
        INDArray indArray = output.linearView();
        List<String> labels = wordToIndex.getLabels();
        Map<String, Float> top10WordsMap = new HashMap<>(labels.size());
        List<String> top10Words = new ArrayList<>(topN);
        Preconditions.checkArgument(topN < labels.size());
        Preconditions.checkArgument(labels.size() == indArray.length());
        for (int i = 0; i < indArray.length(); i++) {
            top10WordsMap.put(labels.get(i), indArray.getFloat(i));
        }
        top10WordsMap.entrySet().parallelStream()
                .sorted((e1, e2) -> -e1.getValue().compareTo(e2.getValue()))
                .limit(topN)
                .forEachOrdered(entry -> top10Words.add(entry.getKey()));
        return top10Words;
    }
}
