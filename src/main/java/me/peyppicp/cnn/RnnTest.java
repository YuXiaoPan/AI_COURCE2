package me.peyppicp.cnn;

import com.google.common.base.Preconditions;
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
    private final PTBEvaluation top1;
    private final PTBEvaluation top3;

    public RnnTest(WordVectors wordVectors, WordToIndex wordToIndex, TokenizerFactory tokenizerFactory, MultiLayerNetwork ptbModel, ComputationGraph emojiModel) {
        this.wordVectors = wordVectors;
        this.wordToIndex = wordToIndex;
        this.tokenizerFactory = tokenizerFactory;
        this.ptbModel = ptbModel;
        this.emojiModel = emojiModel;
        this.word2vecSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
        this.top1 = new PTBEvaluation();
        this.top3 = new PTBEvaluation();
    }

    public String generateTokensFromStr(String sentence) {
        ptbModel.rnnClearPreviousState();
        List<String> tokens = tokenizerFactory.create(sentence).getTokens();
        Preconditions.checkArgument(tokens.size() > 1);
        String firstWord = tokens.get(0);
        INDArray inputArray = Nd4j.zeros(1, word2vecSize);
        INDArray firstVector = wordVectors.getWordVectorMatrix(firstWord);
        inputArray.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all()}, firstVector);
        INDArray firstOutput = ptbModel.rnnTimeStep(inputArray);
        List<String> topNWords = findTopNWords(firstOutput, 10);
        System.out.println(topNWords);
        for (String token : tokens) {
            if (token.equalsIgnoreCase(topNWords.get(0))) {
                top1.tpPlusOne();
            }else{
//                top1.
            }
        }
        return null;
    }

    public List<String> findTopNWords(INDArray output,int topN) {
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
