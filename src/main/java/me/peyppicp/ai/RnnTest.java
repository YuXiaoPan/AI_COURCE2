package me.peyppicp.ai;

import com.vdurmont.emoji.EmojiManager;
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

    private final WordVectors ptbWordVectors;
    private final WordVectors cnnWordVectors;
    private final WordToIndex wordToIndex;
    private final EmojiToIndex emojiToIndex;
    private final TokenizerFactory tokenizerFactory;
    private final MultiLayerNetwork ptbModel;
    private final ComputationGraph cnnModel;
    private final int ptbWord2VecSize;
    //    private final int cnnWord2VecSize;
    private final PTBEvaluation evaluation;
    private final int numberSteps;

    public RnnTest(WordVectors ptbWordVectors, WordVectors cnnWordVectors,
                   WordToIndex wordToIndex, EmojiToIndex emojiToIndex,
                   TokenizerFactory tokenizerFactory, MultiLayerNetwork ptbModel,
                   ComputationGraph cnnModel, int numberSteps) {
        this.ptbWordVectors = ptbWordVectors;
        this.cnnWordVectors = cnnWordVectors;
        this.wordToIndex = wordToIndex;
        this.emojiToIndex = emojiToIndex;
        this.tokenizerFactory = tokenizerFactory;
        this.ptbModel = ptbModel;
        this.cnnModel = cnnModel;
        this.ptbWord2VecSize = ptbWordVectors.getWordVector(ptbWordVectors.vocab().wordAtIndex(0)).length;
//        this.cnnWord2VecSize = cnnWordVectors.getWordVector(cnnWordVectors.vocab().wordAtIndex(0)).length;
        this.evaluation = PTBEvaluation.getInstance();
        this.numberSteps = numberSteps;
    }

    /**
     * 根据输入语句进行预测并统计
     *
     * @param sentence
     * @param topN
     */
    public void generateTokensFromStr(String sentence, int topN) {
        ptbModel.rnnClearPreviousState();
//        拿出emoji
//        List<String> extractEmojis = EmojiParser.extractEmojis(sentence)
//                .stream().distinct().collect(Collectors.toList());
//        删除emoji
//        sentence = EmojiParser.removeAllEmojis(sentence);
//        分词，将所有输入小写并且去除标点符号
        List<String> tokens = tokenizerFactory.create(sentence).getTokens();
        if (tokens.size() <= 1) {
            return;
        }

//        evaluation.plusTotalNumber(tokens.size());
        /*
        构建输入
        如果在第i个词之前有numberStep个词，则从i-numberStep开始重新构建输入
        否则直接计算准确率
         */
        for (int i = 0, j = 1; i < tokens.size() && j < tokens.size(); i++, j++) {
//            int threshold = tokens.size() - numberSteps;
//            if (threshold > 0 && i > numberSteps) {
//                ptbModel.rnnClearPreviousState();
//                for (int k = i + 1 - numberSteps; k < i; k++) {
//                    String previousToken = tokens.get(k);
//                    INDArray zeros = Nd4j.zeros(1, ptbWord2VecSize);
//                    INDArray vectorMatrix;
//                    if (hasToken(previousToken)) {
//                        vectorMatrix = ptbWordVectors.getWordVectorMatrix(previousToken);
//                    } else {
//                        vectorMatrix = ptbWordVectors.getWordVectorMatrix("<unknown>");
//                    }
//                    zeros.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all()}, vectorMatrix);
//                    ptbModel.rnnTimeStep(zeros);
//                }
//            }
            if (i % 5 == 0) {
                ptbModel.rnnClearPreviousState();
            }
            String currentToken = tokens.get(i);
            String nextToken = tokens.get(j);
            INDArray zeros = Nd4j.zeros(1, ptbWord2VecSize);
            INDArray vectorMatrix;
            if (hasToken(currentToken)) {
                vectorMatrix = ptbWordVectors.getWordVectorMatrix(currentToken);
            } else {
                vectorMatrix = ptbWordVectors.getWordVectorMatrix("<unknown>");
            }
            zeros.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all()}, vectorMatrix);
            INDArray output = ptbModel.rnnTimeStep(zeros);
//            获得top3列表
            List<String> top50words = findTopNWords(output, topN)
                    .stream()
                    .filter(s -> !"<unknown>".equals(s))
                    .limit(topN)
                    .collect(Collectors.toList());
//            System.out.println(top50words);
            List<String> top3words = top50words.stream().limit(3).collect(Collectors.toList());
            if (EmojiManager.isEmoji(nextToken)) { //emoji
                if (top3words.get(0).equals(nextToken)) {
                    evaluation.plusEmojiTop1Correct();
                }
                if (top3words.contains(nextToken)) {
                    evaluation.plusEmojiTop3Correct();
                }
                evaluation.plusEmojiTotalNumber();
            } else {
                //单词预测
                if (top3words.get(0).equals(nextToken)) {
                    evaluation.plusTop1Correct();
                }
                if (top3words.contains(nextToken)) {
                    evaluation.plusTop3Correct();
                }
                evaluation.plusTotalNumber();
            }
            System.out.println("Index:" + i + ", finish rate:" + (i / (tokens.size() * 1.0f)) * 100 + "%, top1:" +
                    evaluation.getCorrectTop1Rate() * 100 + "%, top3:" + evaluation.getCorrectTop3Rate() * 100 + "%, " +
                    "emoji:top1:" + evaluation.getCorrectEmojiTop1Rate()*100 + "%, top3:" + evaluation.getCorrectEmojiTop3Rate() *100+
                    "%, Current token:" + currentToken + ", nextToken:" + nextToken + ".");
        }
//        evaluateEmojis(tokens, extractEmojis);
//        evaluation.plusEmojiTotalNumber(extractEmojis.size());
    }

    public boolean hasToken(String token) {
        return ptbWordVectors.hasWord(token) && -1 != wordToIndex.getIndex(token);
    }

    public void evaluateEmojis(List<String> tokens, List<String> emojis) {
//        INDArray input = Nd4j.zeros(1, 1, tokens.size(), cnnWord2VecSize);
        INDArray input = Nd4j.zeros(1, 1, tokens.size(), 0);
        int i = 0;
        for (String token : tokens) {
            INDArray tokenVector = cnnWordVectors.getWordVectorMatrix(token);
            input.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(i++), NDArrayIndex.all()}, tokenVector);
        }
        INDArray output = cnnModel.output(input)[0];
        List<String> emojiTopN = new ArrayList<>();
        Map<String, Float> scoreMap = new HashMap<>();
        List<String> emojisIndex = new ArrayList<>(emojiToIndex.getEmojiToIndexMap().keySet());
        for (int j = 0; j < output.length(); j++) {
            scoreMap.put(emojisIndex.get(j), output.getFloat(j));
        }
        scoreMap.entrySet().stream().sorted(((o1, o2) -> -o1.getValue().compareTo(o2.getValue())))
                .forEachOrdered(entry -> emojiTopN.add(entry.getKey()));
        List<String> emojiTop3 = emojiTopN.stream().limit(3).collect(Collectors.toList());
        for (String emoji : emojis) {
            if (emojiTop3.contains(emoji)) {
                evaluation.plusEmojiTop3Correct();
            }
        }
        for (String emoji : emojis) {
            if (emojiTop3.get(0).equals(emoji)) {
                evaluation.plusEmojiTop1Correct();
            }
        }
    }

    /**
     * 根据模型的输出结果获得对应的token
     *
     * @param output
     * @param topN
     * @return
     */
    public List<String> findTopNWords(INDArray output, int topN) {
        INDArray indArray = output.linearView();
        List<String> labels = wordToIndex.getLabels();
        Map<String, Float> top10WordsMap = new HashMap<>(30000);
        List<String> top10Words = new ArrayList<>(topN);
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
