package me.peyppicp.ai;

import me.peyppicp.Utils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author YuXiao Pan
 * @date 2017/12/28
 * @email yuxiao.pan@kikatech.com
 */
public class PTBTestMain {

    public static void main(String[] args) throws IOException {
        String prefix = "";
        String output = "";
        String ptbModelPath = args[0];
        String ptbWord2VecPath = args[1];
        String ptbPairPath = args[2];
//        String cnnModelPath = "";
//        String cnnWord2VecPath = "";
        String testDataPath = args[3];
        MultiLayerNetwork ptbModel = ModelSerializer.restoreMultiLayerNetwork(ptbModelPath);
        WordVectors ptbWordVectors = WordVectorSerializer.readWord2VecModel(ptbWord2VecPath);
        WordToIndex wordToIndex = new WordToIndex(ptbPairPath);

//        ComputationGraph cnnModel = ModelSerializer.restoreComputationGraph(cnnModelPath);
//        Word2Vec cnnWord2Vec = WordVectorSerializer.readWord2VecModel(cnnWord2VecPath);
//        EmojiToIndex emojiToIndex = new EmojiToIndex();

        DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

//        RnnTest rnnTest = new RnnTest(ptbWordVectors, cnnWord2Vec,
//                wordToIndex, emojiToIndex, tokenizerFactory, ptbModel, cnnModel, 3);
        RnnTest rnnTest = new RnnTest(ptbWordVectors, null,
                wordToIndex, null, tokenizerFactory, ptbModel, null, 5);
        List<String> collect = Utils.readLinesFromPath(testDataPath);
        List<String> data = preprocessData(collect, tokenizerFactory);
//        for (String s : collect) {
//            rnnTest.generateTokensFromStr(s, 100);
//        }

        rnnTest.generateTokensFromStr(data, 100);

//        int threadNum = 8;
//        List<List<String>> testDatas = new ArrayList<>();
//        for (int i = 0; i < threadNum; i++) {
//            testDatas.add(new ArrayList<>());
//        }
//        for (int i = 0; i < collect.size(); i++) {
//            int number = i % threadNum;
//            List<String> strings = testDatas.get(number);
//            strings.add(collect.get(i));
//        }
//
//        ExecutorService executorService = Executors.newFixedThreadPool(threadNum);
//        for (int i = 0; i < threadNum; i++) {
//            executorService.submit(new EvaluationRunner(ModelSerializer.restoreMultiLayerNetwork(prefix + "ptb2.txt"),
//                    wordVectors, new WordToIndex(prefix + "pair.txt"), testDatas.get(i), tokenizerFactory));
//        }
//
//        System.out.println("--------------");
//        System.out.println(PTBEvaluation.getInstance().getCorrectTop1Rate());
//        System.out.println(PTBEvaluation.getInstance().getCorrectTop3Rate());
    }

    public static List<String> preprocessData(List<String> testData, TokenizerFactory tokenizerFactory) {
        List<String> allTokens = new ArrayList<>();
        for (String data1 : testData) {
            List<String> tokens = tokenizerFactory.create(data1).getTokens();
            if (tokens.size() > 1) {
                allTokens.addAll(tokens);
            }
        }
        return allTokens;
    }
}

class EvaluationRunner implements Runnable {

    private MultiLayerNetwork multiLayerNetwork;
    private WordVectors wordVectors;
    private WordToIndex wordToIndex;
    private List<String> testData;
    private DefaultTokenizerFactory defaultTokenizerFactory;

    public EvaluationRunner(MultiLayerNetwork multiLayerNetwork, WordVectors wordVectors, WordToIndex wordToIndex, List<String> testData, DefaultTokenizerFactory defaultTokenizerFactory) {
        this.multiLayerNetwork = multiLayerNetwork;
        this.wordVectors = wordVectors;
        this.wordToIndex = wordToIndex;
        this.testData = testData;
        this.defaultTokenizerFactory = defaultTokenizerFactory;
    }

    @Override
    public void run() {
//        RnnTest rnnTest = new RnnTest(wordVectors, wordToIndex, defaultTokenizerFactory, multiLayerNetwork, null, 3);
//        for (String s : testData) {
//            rnnTest.generateTokensFromStr(s, 10);
//        }
    }
}
